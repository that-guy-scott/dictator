import json
import os
import socket
from pathlib import Path
from typing import Any, Generator


def get_runtime_dir() -> Path:
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg)
    return Path(f"/tmp/dictator-{os.getuid()}")


def get_socket_path() -> Path:
    return get_runtime_dir() / "dictator.sock"


def get_pid_path() -> Path:
    return get_runtime_dir() / "dictator.pid"


def ensure_runtime_dir() -> None:
    d = get_runtime_dir()
    d.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("XDG_RUNTIME_DIR"):
        os.chmod(d, 0o700)


def send_message(sock: socket.socket, msg: dict[str, Any]) -> None:
    data = json.dumps(msg) + "\n"
    sock.sendall(data.encode())


def receive_messages(sock: socket.socket) -> Generator[dict[str, Any], None, None]:
    buf = b""
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        except (ConnectionResetError, BrokenPipeError, OSError):
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line:
                yield json.loads(line.decode())


def connect() -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(get_socket_path()))
    return sock


def is_daemon_alive() -> bool:
    pid_path = get_pid_path()
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        return False


def read_pid() -> int | None:
    pid_path = get_pid_path()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def cleanup_stale_files() -> None:
    sock_path = get_socket_path()
    pid_path = get_pid_path()
    if sock_path.exists():
        sock_path.unlink()
    if pid_path.exists():
        pid_path.unlink()
