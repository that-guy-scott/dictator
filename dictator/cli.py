import argparse
import signal
import sys

from dictator.config import load_config
from dictator.daemon import start_daemon
from dictator.protocol import (
    connect,
    is_daemon_alive,
    read_pid,
    receive_messages,
    send_message,
)


def cmd_start(args: argparse.Namespace) -> None:
    config = load_config()
    start_daemon(config)


def cmd_stop(args: argparse.Namespace) -> None:
    if not is_daemon_alive():
        print("Daemon is not running.")
        return

    try:
        sock = connect()
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        print(f"Cannot connect to daemon: {e}", file=sys.stderr)
        sys.exit(1)

    send_message(sock, {"command": "shutdown"})
    for msg in receive_messages(sock):
        if msg.get("type") == "status" and msg.get("state") == "shutting_down":
            print("Daemon stopping...")
            break
    sock.close()

    # Wait for PID to exit
    pid = read_pid()
    if pid:
        import time
        for _ in range(50):  # 5 seconds
            try:
                import os
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                break
    print("Daemon stopped.")


def cmd_record(args: argparse.Namespace) -> None:
    if not is_daemon_alive():
        print("Daemon is not running. Start it with: dictator start", file=sys.stderr)
        sys.exit(1)

    try:
        sock = connect()
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        print(f"Cannot connect to daemon: {e}", file=sys.stderr)
        sys.exit(1)

    # Set a timeout so recv doesn't block forever
    sock.settimeout(2.0)

    send_message(sock, {"command": "record_toggle"})

    try:
        for msg in receive_messages(sock):
            msg_type = msg.get("type")
            if msg_type == "status":
                state = msg.get("state")
                if state == "recording":
                    print("Recording... (Ctrl+C to stop)", file=sys.stderr)
            elif msg_type == "transcript":
                text = msg.get("text", "")
                if text:
                    print(text)
                if msg.get("final"):
                    break
            elif msg_type == "error":
                print(f"Error: {msg.get('message')}", file=sys.stderr)
                break
    except KeyboardInterrupt:
        # Send stop command to daemon
        try:
            stop_sock = connect()
            stop_sock.settimeout(5.0)
            send_message(stop_sock, {"command": "record_toggle"})
            # Drain any final transcripts
            for msg in receive_messages(stop_sock):
                if msg.get("type") == "transcript":
                    text = msg.get("text", "")
                    if text:
                        print(text)
                    if msg.get("final"):
                        break
            stop_sock.close()
        except (BrokenPipeError, ConnectionResetError, OSError, KeyboardInterrupt):
            pass
        print("\nStopped.", file=sys.stderr)

    sock.close()


def cmd_status(args: argparse.Namespace) -> None:
    if not is_daemon_alive():
        print("Daemon is not running.")
        return

    try:
        sock = connect()
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        print(f"Cannot connect to daemon: {e}", file=sys.stderr)
        sys.exit(1)

    send_message(sock, {"command": "status"})
    for msg in receive_messages(sock):
        if msg.get("type") == "status":
            print(f"Daemon status: {msg.get('state')}")
            break
        elif msg.get("type") == "error":
            print(f"Error: {msg.get('message')}", file=sys.stderr)
            break
    sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dictator",
        description="Terminal voice-to-text dictation using Whisper",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start the dictator daemon")
    sub.add_parser("stop", help="Stop the dictator daemon")
    sub.add_parser("record", help="Toggle recording (Ctrl+C to stop)")
    sub.add_parser("status", help="Show daemon status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "record": cmd_record,
        "status": cmd_status,
    }
    commands[args.command](args)
