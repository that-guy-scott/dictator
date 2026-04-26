import datetime
import logging
import logging.handlers
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from dictator.config import Config, load_config
from dictator.protocol import (
    cleanup_stale_files,
    ensure_runtime_dir,
    get_pid_path,
    get_socket_path,
    is_daemon_alive,
    receive_messages,
    send_message,
)
from dictator.recorder import Recorder, Sentinel
from dictator.transcriber import Transcriber
from dictator.vad import VadChunker

log = logging.getLogger("dictator")

DATA_DIR = Path.home() / ".local" / "share" / "dictator"
HISTORY_PATH = DATA_DIR / "history.log"
LOG_PATH = DATA_DIR / "daemon.log"


def _setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    ))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)


def _log_history(text: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with open(HISTORY_PATH, "a") as f:
        f.write(f"[{ts}] {text}\n")


def _output_text(text: str, mode: str) -> None:
    """Output transcribed text via the configured method."""
    log.info("Output mode='%s', text='%s'", mode, text[:80])

    if mode == "type":
        # Try typing tools in order of preference for each display server
        # ydotool: works on any Wayland (kernel-level input), needs ydotoold
        ydotool = shutil.which("ydotool")
        if ydotool:
            log.info("Typing via ydotool")
            result = subprocess.run([ydotool, "type", "--", text], check=False, capture_output=True)
            if result.returncode == 0:
                return
            log.warning("ydotool failed (rc=%d): %s", result.returncode, result.stderr.decode())

        # wtype: works on wlroots Wayland compositors (Sway, Hyprland)
        wtype = shutil.which("wtype")
        if wtype:
            log.info("Typing via wtype")
            result = subprocess.run([wtype, "--", text], check=False, capture_output=True)
            if result.returncode == 0:
                return
            log.warning("wtype failed (rc=%d): %s", result.returncode, result.stderr.decode())

        # xdotool: works on X11
        xdotool = shutil.which("xdotool")
        if xdotool:
            log.info("Typing via xdotool")
            result = subprocess.run([xdotool, "type", "--clearmodifiers", "--", text], check=False, capture_output=True)
            if result.returncode == 0:
                return
            log.warning("xdotool failed (rc=%d): %s", result.returncode, result.stderr.decode())

        log.warning("All typing tools failed. Falling back to clipboard.")
        mode = "clipboard"

    if mode == "clipboard":
        # Copy to clipboard: try wl-copy (Wayland), then xclip (X11)
        wl_copy = shutil.which("wl-copy")
        if wl_copy:
            log.info("Copying to clipboard via wl-copy")
            subprocess.run([wl_copy, "--", text], check=False)
            return
        xclip = shutil.which("xclip")
        if xclip:
            subprocess.run([xclip, "-selection", "clipboard"], input=text.encode(), check=False)
            return
        log.warning("Output mode 'clipboard' but neither wl-copy nor xclip found.")


class Daemon:
    def __init__(self, config: Config, ready_fd: int | None = None) -> None:
        self.config = config
        self._shutdown = threading.Event()
        self._recording_lock = threading.Lock()
        self._recording_client: socket.socket | None = None
        self._recorder: Recorder | None = None
        self._processing_thread: threading.Thread | None = None
        self._audio_queue: queue.Queue | None = None

        # Load the whisper model (this is the slow part)
        log.info("Initializing transcriber...")
        self.transcriber = Transcriber(config)

        # Set up socket server
        ensure_runtime_dir()
        sock_path = get_socket_path()
        if sock_path.exists():
            sock_path.unlink()

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(sock_path))
        os.chmod(str(sock_path), 0o600)
        self._server.listen(5)
        self._server.settimeout(1.0)  # So we can check shutdown flag

        # Write PID
        pid_path = get_pid_path()
        pid_path.write_text(str(os.getpid()))

        log.info("Daemon ready (PID %d, socket %s)", os.getpid(), sock_path)

        # Signal to parent that we're ready
        if ready_fd is not None:
            os.write(ready_fd, b"R")
            os.close(ready_fd)

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        log.info("Daemon accept loop started.")
        while not self._shutdown.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._shutdown.is_set():
                    break
                raise

            try:
                self._handle_connection(conn)
            except Exception:
                log.exception("Error handling connection")
                conn.close()

        self._cleanup()

    def _signal_handler(self, signum, frame) -> None:
        log.info("Received signal %d, shutting down...", signum)
        self._shutdown.set()

    def _handle_connection(self, conn: socket.socket) -> None:
        for msg in receive_messages(conn):
            cmd = msg.get("command")
            if cmd == "status":
                self._handle_status(conn)
                conn.close()
                return
            elif cmd == "shutdown":
                self._handle_shutdown(conn)
                return
            elif cmd == "record_toggle":
                self._handle_record_toggle(conn)
                return
            elif cmd == "record_cancel":
                self._handle_record_cancel(conn)
                conn.close()
                return
            else:
                send_message(conn, {"type": "error", "message": f"Unknown command: {cmd}"})
                conn.close()
                return
        conn.close()

    def _handle_status(self, conn: socket.socket) -> None:
        if self._recording_client is not None:
            state = "recording"
        else:
            state = "idle"
        send_message(conn, {"type": "status", "state": state})

    def _handle_shutdown(self, conn: socket.socket) -> None:
        log.info("Shutdown requested by client.")
        self._stop_recording()
        send_message(conn, {"type": "status", "state": "shutting_down"})
        conn.close()
        self._shutdown.set()

    def _handle_record_toggle(self, conn: socket.socket) -> None:
        with self._recording_lock:
            if self._recording_client is not None:
                # Already recording — stop
                log.info("Stopping recording (toggle off).")
                self._stop_recording()
                conn.close()
            else:
                # Start recording
                log.info("Starting recording (toggle on).")
                self._start_recording(conn)

    def _handle_record_cancel(self, conn: socket.socket) -> None:
        with self._recording_lock:
            if self._recording_client is not None:
                log.info("Cancelling recording.")
                self._cancel_recording()
                send_message(conn, {"type": "status", "state": "cancelled"})
            else:
                send_message(conn, {"type": "error", "message": "Not recording"})

    def _start_recording(self, client_conn: socket.socket) -> None:
        self._audio_queue = queue.Queue(maxsize=500)
        self._recording_client = client_conn

        try:
            self._recorder = Recorder(self.config, self._audio_queue)
            self._recorder.start()
        except Exception as e:
            log.exception("Failed to start audio capture")
            try:
                send_message(client_conn, {"type": "error", "message": str(e)})
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            self._recording_client = None
            self._recorder = None
            client_conn.close()
            return

        send_message(client_conn, {"type": "status", "state": "recording"})

        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
        )
        self._processing_thread.start()

    def _stop_recording(self) -> None:
        if self._recorder:
            self._recorder.stop()
        if self._processing_thread:
            self._processing_thread.join(timeout=30)
            self._processing_thread = None
        self._recorder = None
        self._audio_queue = None

    def _cancel_recording(self) -> None:
        if self._recorder:
            self._recorder.cancel()
        if self._processing_thread:
            self._processing_thread.join(timeout=10)
            self._processing_thread = None
        if self._recording_client:
            try:
                self._recording_client.close()
            except OSError:
                pass
            self._recording_client = None
        self._recorder = None
        self._audio_queue = None

    def _processing_loop(self) -> None:
        client = self._recording_client
        previous_prompt: str | None = None
        clipboard_text: str = ""

        output_mode = self.config.output.mode

        def on_segment(segment: np.ndarray) -> None:
            nonlocal previous_prompt, clipboard_text
            log.info("Transcribing segment (%.2fs)...", len(segment) / self.config.audio.sample_rate)
            text = self.transcriber.transcribe(segment, prompt=previous_prompt)
            if text:
                previous_prompt = text
                _log_history(text)

                # Output via configured method
                if output_mode == "clipboard":
                    # Clipboard: accumulate full transcript so user can paste everything
                    clipboard_text = text if not clipboard_text else clipboard_text + "\n" + text
                    _output_text(clipboard_text, "clipboard")
                elif output_mode == "type":
                    # Type: send just this phrase (it's typed live)
                    _output_text(text + " ", "type")

                # Only echo transcript text to the CLI in stdout mode.
                if output_mode == "stdout":
                    try:
                        send_message(client, {"type": "transcript", "text": text, "final": False})
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        log.warning("Client disconnected during transcription.")
                        if self._recorder:
                            self._recorder.cancel()

        try:
            vad = VadChunker(self.config, on_segment=on_segment)
            chunk_count = 0

            while True:
                try:
                    item = self._audio_queue.get(timeout=1.0)
                except queue.Empty:
                    if self._shutdown.is_set():
                        break
                    continue

                if item is Sentinel.CANCEL:
                    vad.reset()
                    break

                if item is Sentinel.STOP:
                    vad.flush()
                    break

                chunk_count += 1
                if chunk_count == 1:
                    log.info("First audio chunk received (len=%d, max=%.4f)", len(item), float(item.max()))
                elif chunk_count % 500 == 0:
                    log.info("Audio chunks received: %d (latest max=%.4f)", chunk_count, float(item.max()))

                vad.feed(item)
        except Exception:
            log.exception("Processing thread crashed")

        # Send final message and close client connection
        if client:
            try:
                send_message(client, {"type": "transcript", "text": "", "final": True})
                client.close()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        self._recording_client = None
        log.info("Processing loop finished.")

    def _cleanup(self) -> None:
        log.info("Cleaning up...")
        self._stop_recording()
        self._server.close()
        cleanup_stale_files()
        log.info("Daemon shut down.")


def start_daemon(config: Config | None = None) -> None:
    if is_daemon_alive():
        print("Daemon is already running.")
        sys.exit(0)

    # Clean up stale files from a previous crash
    cleanup_stale_files()

    config = config or load_config()

    read_fd, write_fd = os.pipe()

    pid = os.fork()
    if pid > 0:
        # Parent: wait for ready signal
        os.close(write_fd)
        read_file = os.fdopen(read_fd, "rb")
        # Wait up to 120 seconds for model loading
        import select
        ready, _, _ = select.select([read_file], [], [], 120)
        if ready:
            data = read_file.read(1)
            if data == b"R":
                print("Daemon started (PID %d). Model loaded." % pid)
            else:
                print("Daemon failed to start.", file=sys.stderr)
                sys.exit(1)
        else:
            print("Timed out waiting for daemon to load model.", file=sys.stderr)
            sys.exit(1)
        read_file.close()
    else:
        # Child: become daemon
        os.close(read_fd)
        os.setsid()

        # Redirect stdio to /dev/null
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)

        _setup_logging()
        try:
            daemon = Daemon(config, ready_fd=write_fd)
            daemon.run()
        except Exception:
            log.exception("Daemon crashed")
            cleanup_stale_files()
            sys.exit(1)
