from __future__ import annotations

import logging
from collections.abc import Generator

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Loads and caches both Whisper models at startup."""

    def __init__(self, device: str = "cuda", compute_type: str = "float16"):
        self.device = device
        self.compute_type = compute_type
        self._streaming_model: WhisperModel | None = None
        self._final_model: WhisperModel | None = None

    def get_streaming_model(self) -> WhisperModel:
        """distil-whisper/distil-large-v3 — optimized for speed."""
        if self._streaming_model is None:
            logger.info("Loading streaming model (distil-large-v3)...")
            self._streaming_model = WhisperModel(
                "distil-large-v3",
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=1,
            )
            logger.info("Streaming model loaded")
        return self._streaming_model

    def get_final_model(self) -> WhisperModel:
        """whisper-large-v3 — highest accuracy for final pass."""
        if self._final_model is None:
            logger.info("Loading final model (large-v3)...")
            self._final_model = WhisperModel(
                "large-v3",
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=1,
            )
            logger.info("Final model loaded")
        return self._final_model


class DualPassTranscriber:
    """Two-pass ASR strategy.

    Pass 1 (Streaming): distil-whisper/distil-large-v3
      - Processes audio in 30-second chunks
      - Emits partial transcripts as they complete
      - Lower accuracy but fast turnaround

    Pass 2 (Final): whisper-large-v3
      - Processes the full audio file
      - Produces definitive transcript
      - Replaces streaming partials
    """

    CHUNK_DURATION = 30.0  # seconds per streaming chunk

    def __init__(
        self,
        streaming_model: WhisperModel,
        final_model: WhisperModel,
    ):
        self.streaming_model = streaming_model
        self.final_model = final_model

    def streaming_pass(
        self, audio: np.ndarray, sr: int = 44100
    ) -> Generator[dict, None, None]:
        """Yield partial transcript segments from distil-whisper."""
        chunk_samples = int(self.CHUNK_DURATION * sr)
        offset = 0.0

        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < sr:  # skip chunks < 1 second
                continue

            segments, info = self.streaming_model.transcribe(
                chunk,
                language=None,
                beam_size=1,
                best_of=1,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
            )

            for seg in segments:
                words = []
                for w in seg.words or []:
                    words.append({
                        "word": w.word,
                        "start": round(w.start + offset, 3),
                        "end": round(w.end + offset, 3),
                        "probability": round(w.probability, 4),
                    })

                yield {
                    "t_start": round(seg.start + offset, 3),
                    "t_end": round(seg.end + offset, 3),
                    "text": seg.text.strip(),
                    "confidence": round(seg.avg_logprob, 4) if seg.avg_logprob else None,
                    "words": words,
                    "is_final": False,
                }

            offset += self.CHUNK_DURATION

    def final_pass(self, audio: np.ndarray, sr: int = 44100) -> list[dict]:
        """Full-file transcription with whisper-large-v3."""
        segments, info = self.final_model.transcribe(
            audio,
            language=None,
            beam_size=5,
            best_of=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
            condition_on_previous_text=True,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        results = []
        for seg in segments:
            words = []
            for w in seg.words or []:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                })

            results.append({
                "t_start": round(seg.start, 3),
                "t_end": round(seg.end, 3),
                "text": seg.text.strip(),
                "confidence": round(seg.avg_logprob, 4) if seg.avg_logprob else None,
                "words": words,
                "is_final": True,
            })

        return results
