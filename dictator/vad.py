import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dictator.config import Config

log = logging.getLogger(__name__)

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
CACHE_DIR = Path.home() / ".cache" / "dictator"
MODEL_PATH = CACHE_DIR / "silero_vad.onnx"

# Silero VAD expects 512 new samples + 64 context samples at 16kHz
CHUNK_SAMPLES = 512
CONTEXT_SIZE = 64  # 16kHz context window
SPEECH_THRESHOLD = 0.35


def _ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Silero VAD model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info("Silero VAD model saved to %s", MODEL_PATH)
    return MODEL_PATH


class VadChunker:
    def __init__(
        self,
        config: Config,
        on_segment: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        model_path = _ensure_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.sample_rate = config.audio.sample_rate
        self.pause_threshold = config.vad.pause_threshold
        self.min_chunk_length = config.vad.min_chunk_length
        self.max_segment_length = config.vad.max_segment_length
        self.on_segment = on_segment

        # Samples of silence needed before we consider speech ended
        self._pause_samples = int(self.pause_threshold * self.sample_rate)
        self._max_segment_samples = int(self.max_segment_length * self.sample_rate)

        self._reset_state()

    def _reset_state(self) -> None:
        # ONNX model state (LSTM hidden/cell)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(self.sample_rate, dtype=np.int64)

        # Context window: last 64 samples from previous chunk
        self._context = np.zeros((1, CONTEXT_SIZE), dtype=np.float32)

        # Speech tracking
        self._is_speaking = False
        self._speech_buffer: list[np.ndarray] = []
        self._buffered_samples: int = 0
        self._silence_count = 0  # samples of silence since last speech

        # Leftover audio that didn't fill a full 512-sample chunk
        self._leftover = np.array([], dtype=np.float32)

    def _infer(self, chunk: np.ndarray) -> float:
        # Prepend context to chunk: model expects [context(64) + audio(512)] = 576 samples
        chunk_2d = chunk.reshape(1, -1)
        input_tensor = np.concatenate([self._context, chunk_2d], axis=1).astype(np.float32)

        ort_inputs = {
            "input": input_tensor,
            "state": self._state,
            "sr": self._sr,
        }
        output, new_state = self.session.run(None, ort_inputs)
        self._state = new_state

        # Update context with last 64 samples of the full input
        self._context = input_tensor[:, -CONTEXT_SIZE:]

        return float(output[0][0])

    def feed(self, audio: np.ndarray) -> None:
        audio = audio.astype(np.float32).ravel()

        # Prepend any leftover from previous call
        if len(self._leftover) > 0:
            audio = np.concatenate([self._leftover, audio])
            self._leftover = np.array([], dtype=np.float32)

        offset = 0
        while offset + CHUNK_SAMPLES <= len(audio):
            chunk = audio[offset : offset + CHUNK_SAMPLES]
            offset += CHUNK_SAMPLES
            self._process_chunk(chunk)

        # Save leftover
        if offset < len(audio):
            self._leftover = audio[offset:]

    def _process_chunk(self, chunk: np.ndarray) -> None:
        prob = self._infer(chunk)

        if prob >= SPEECH_THRESHOLD:
            # Speech detected
            if not self._is_speaking:
                self._is_speaking = True
                self._silence_count = 0
                log.info("Speech started (prob=%.4f)", prob)
            self._speech_buffer.append(chunk)
            self._buffered_samples += len(chunk)
            self._silence_count = 0

            # Force-emit if segment exceeds the length cap
            if self._buffered_samples >= self._max_segment_samples:
                segment = np.concatenate(self._speech_buffer)
                duration = len(segment) / self.sample_rate
                self._speech_buffer = []
                self._buffered_samples = 0
                self._silence_count = 0
                # Keep _is_speaking = True — speech is still active
                log.info("Forcing segment split at %.2fs (cap=%.1fs)", duration, self.max_segment_length)
                if self.on_segment:
                    self.on_segment(segment)
        else:
            if self._is_speaking:
                # Still in a speech region, counting silence
                self._speech_buffer.append(chunk)
                self._buffered_samples += len(chunk)
                self._silence_count += CHUNK_SAMPLES
                if self._silence_count >= self._pause_samples:
                    self._emit_segment()
            # else: silence outside of speech, ignore

    def _emit_segment(self) -> None:
        if not self._speech_buffer:
            self._is_speaking = False
            self._silence_count = 0
            return

        segment = np.concatenate(self._speech_buffer)

        # Trim trailing silence that was buffered while waiting for the
        # pause threshold.  Leaving it in causes Whisper to hallucinate
        # repeated text to fill the dead air.
        if self._silence_count > 0:
            trim = min(self._silence_count, len(segment) - CHUNK_SAMPLES)
            if trim > 0:
                segment = segment[:-trim]

        duration = len(segment) / self.sample_rate

        self._is_speaking = False
        self._speech_buffer = []
        self._buffered_samples = 0
        self._silence_count = 0

        if duration < self.min_chunk_length:
            log.debug("Discarding short segment (%.2fs < %.2fs)", duration, self.min_chunk_length)
            return

        log.info("Speech segment ready: %.2fs", duration)
        if self.on_segment:
            self.on_segment(segment)

    def flush(self) -> None:
        if self._speech_buffer:
            # Include any leftover audio
            if len(self._leftover) > 0:
                self._speech_buffer.append(self._leftover)
                self._leftover = np.array([], dtype=np.float32)
            self._emit_segment()

    def reset(self) -> None:
        self._reset_state()
