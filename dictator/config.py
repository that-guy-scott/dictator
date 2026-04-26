import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys
import tomllib

log = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    device: str = "default"
    sample_rate: int = 16000


@dataclass
class WhisperConfig:
    model: str = "large-v3"
    language: str = "en"


@dataclass
class VadConfig:
    pause_threshold: float = 0.5
    min_chunk_length: float = 1.0
    max_segment_length: float = 30.0


@dataclass
class OutputConfig:
    mode: str = "stdout"  # stdout | clipboard | type


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


CONFIG_PATH = Path.home() / ".config" / "dictator" / "config.toml"


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    config = Config()

    if not path.exists():
        return config

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Error parsing config file {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if "audio" in data:
        for key in ("device", "sample_rate"):
            if key in data["audio"]:
                setattr(config.audio, key, data["audio"][key])

    if "whisper" in data:
        for key in ("model", "language"):
            if key in data["whisper"]:
                setattr(config.whisper, key, data["whisper"][key])

    if "vad" in data:
        for key in ("pause_threshold", "min_chunk_length", "max_segment_length"):
            if key in data["vad"]:
                setattr(config.vad, key, float(data["vad"][key]))

    if config.vad.max_segment_length <= config.vad.min_chunk_length:
        clamped = config.vad.min_chunk_length * 2
        log.warning(
            "max_segment_length (%.1f) must be > min_chunk_length (%.1f); clamping to %.1f",
            config.vad.max_segment_length,
            config.vad.min_chunk_length,
            clamped,
        )
        config.vad.max_segment_length = clamped

    if "output" in data:
        if "mode" in data["output"]:
            config.output.mode = data["output"]["mode"]

    return config
