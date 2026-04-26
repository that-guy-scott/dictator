import logging
import numpy as np
from faster_whisper import WhisperModel

from dictator.config import Config

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, config: Config) -> None:
        model_name = config.whisper.model
        log.info("Loading Whisper model '%s' onto GPU...", model_name)
        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16",
        )
        self.language = config.whisper.language if config.whisper.language != "auto" else None
        log.info("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=prompt,
            vad_filter=False,  # we handle VAD ourselves
        )
        return "".join(seg.text for seg in segments).strip()
