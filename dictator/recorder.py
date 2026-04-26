import logging
import queue
from enum import Enum, auto

import numpy as np
import sounddevice as sd

from dictator.config import Config

log = logging.getLogger(__name__)


class Sentinel(Enum):
    STOP = auto()
    CANCEL = auto()


class Recorder:
    def __init__(self, config: Config, audio_queue: queue.Queue) -> None:
        self.device = config.audio.device if config.audio.device != "default" else None
        self.sample_rate = config.audio.sample_rate
        self.audio_queue = audio_queue
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        # Copy the data — indata buffer is reused by sounddevice
        self.audio_queue.put(indata[:, 0].copy())

    def start(self) -> None:
        log.info("Starting audio capture (device=%s, rate=%d)", self.device, self.sample_rate)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.audio_queue.put(Sentinel.STOP)
        log.info("Audio capture stopped.")

    def cancel(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.audio_queue.put(Sentinel.CANCEL)
        log.info("Audio capture cancelled.")
