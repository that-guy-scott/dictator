# Dictator — Terminal Voice-to-Text Tool

## Overview

A terminal voice-to-text dictation tool for Linux. A background daemon keeps a Whisper model warm in GPU memory. A thin CLI client triggers recording, and transcribed text streams to stdout phrase-by-phrase using VAD-based chunking.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      dictator daemon                         │
│                                                              │
│  ┌────────────┐   ┌────────────┐   ┌───────────────────┐    │
│  │  Audio      │   │  Silero    │   │  faster-whisper   │    │
│  │  recorder   │──▶│  VAD       │──▶│  (model warm      │    │
│  │ (sounddevice│   │  (chunker) │   │   in VRAM)        │    │
│  └────────────┘   └────────────┘   └───────────────────┘    │
│                                                              │
│  Unix socket server (listens for CLI commands)               │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────────────┐
│                      dictator CLI                            │
│                                                              │
│  dictator start          Launch daemon, load model           │
│  dictator stop           Shut down daemon, free VRAM         │
│  dictator record         Toggle recording on/off             │
│  dictator status         Show daemon state (idle/recording)  │
└──────────────────────────────────────────────────────────────┘
```

## Audio Pipeline

```
Mic → sounddevice stream → Ring buffer → Silero VAD → Speech segments → Whisper → stdout
```

1. Audio is captured continuously from the mic while recording is active.
2. Silero VAD runs on the audio stream (CPU, near-zero overhead) and detects speech boundaries.
3. When a pause is detected (configurable threshold), the speech segment is sent to Whisper.
4. Whisper transcribes the segment. The previous segment's text is fed as `initial_prompt` for cross-chunk context.
5. Transcribed text is sent back to the CLI client over the unix socket and printed to stdout.

## Recording Modes

- **Toggle:** Press hotkey once to start recording, press again to stop. Preferred for RSI.
- **Push-to-talk:** Hold hotkey to record, release to stop.
- Mode is configurable. Both bindings can be active simultaneously.

## Technology Stack

| Component        | Library / Tool      | Notes                                         |
|------------------|---------------------|-----------------------------------------------|
| Language         | Python 3.11+        |                                                |
| Audio capture    | sounddevice         | PortAudio bindings, simple API                 |
| VAD              | Silero VAD          | Lightweight, accurate, runs on CPU             |
| Transcription    | faster-whisper       | CTranslate2-based, ~4x faster than openai-whisper |
| IPC              | Unix domain socket   | Daemon ↔ CLI communication                   |
| Config           | TOML                | `tomllib` (stdlib 3.11+) for reading           |
| CLI              | argparse            | No external dependency needed                  |

## Configuration

Location: `~/.config/dictator/config.toml`

```toml
[audio]
device = "default"            # Mic device name or "default"
sample_rate = 16000           # Whisper expects 16kHz

[whisper]
model = "large-v3"            # tiny | base | small | medium | large-v3
language = "en"               # ISO 639-1 code, or "auto"

[vad]
pause_threshold = 0.5         # Seconds of silence before a chunk is sent
min_chunk_length = 1.0        # Minimum speech segment length (seconds)

[keybinds]
toggle = "ctrl+shift+d"       # Toggle recording on/off
push_to_talk = "ctrl+shift+space"  # Hold to record

[output]
mode = "stdout"               # stdout | clipboard | type
```

## Daemon Lifecycle

- `dictator start` — Forks a background process. Loads the Whisper model into GPU memory. Creates a unix socket at `/tmp/dictator.sock` (or XDG runtime dir). Writes PID to `/tmp/dictator.pid`.
- `dictator stop` — Sends shutdown command over the socket. Daemon unloads model, cleans up socket and PID file.
- `dictator status` — Queries daemon state: not running, idle, recording, or transcribing.
- `dictator record` — Sends toggle command. Daemon starts/stops audio capture. Transcribed text streams back over the socket to the CLI, which prints it to stdout.

## Silence Trimming

Leading and trailing silence is trimmed from each speech segment before sending to Whisper. This reduces inference time and improves accuracy.

## Transcription Context

To maintain coherence across chunks, the daemon feeds the previous chunk's transcription as Whisper's `initial_prompt`. This helps with:
- Consistent capitalization
- Proper nouns and technical terms
- Natural sentence flow

## Cancel Recording

A recording can be cancelled (discarded without transcription) by pressing `Escape` or a configurable cancel key. Important for avoiding unnecessary transcription on accidental triggers.

## Transcription History

All transcriptions are logged to `~/.local/share/dictator/history.log` with timestamps. Low effort, useful for reviewing past dictations.

## Error Handling

- **No microphone found:** Daemon logs error, CLI reports it on `dictator start`.
- **Model download:** If the model isn't cached, `faster-whisper` downloads it on first `dictator start`. Progress is shown.
- **Daemon not running:** CLI commands (other than `start`) print a clear message if the daemon isn't running.
- **GPU out of memory:** Daemon falls back to a smaller model and warns the user.

## Dependencies

```
faster-whisper
sounddevice
silero-vad (via torch hub or bundled)
torch (CUDA)
```

## File Structure

```
dictator/
├── dictator/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point and argument parsing
│   ├── daemon.py           # Daemon process, socket server, lifecycle
│   ├── recorder.py         # Audio capture with sounddevice
│   ├── vad.py              # Silero VAD integration and chunking
│   ├── transcriber.py      # faster-whisper model loading and inference
│   ├── config.py           # Config file loading and defaults
│   └── protocol.py         # Unix socket message protocol
├── pyproject.toml
├── tech-spec.md
└── prompt.md
```

## Hardware Target

- GPU: NVIDIA RTX 3090 (24GB VRAM)
- OS: Linux (Fedora)
- Model: `large-v3` via faster-whisper fits comfortably in VRAM
