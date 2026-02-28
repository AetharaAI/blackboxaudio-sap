"""SAP Flamingo Worker — Deep music understanding via NVIDIA Music Flamingo.

Consumes from sap:flamingo:pending, loads audio from MinIO, runs structured
prompts through Music Flamingo (8B), and stores analysis results.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid

import numpy as np
import soundfile as sf

from sap_common.config import settings
from sap_common.db import async_session_factory
from sap_common.minio_client import download_bytes
from sap_common.models import FlamingoAnalysis
from sap_common.valkey_streams import StreamWorker

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("FLAMINGO_MODEL_ID", "nvidia/music-flamingo-2601-hf")

# Structured prompts for music analysis
ANALYSIS_PROMPTS = [
    {
        "key": "overview",
        "prompt": (
            "Describe this track in full detail - tell me the genre, tempo, and key, "
            "then dive into the instruments, production style, and overall mood it creates."
        ),
    },
    {
        "key": "structure",
        "prompt": (
            "Identify the song structure section by section (intro, verse, chorus, bridge, "
            "outro, etc). For each section, note the approximate time range and what "
            "characterizes it musically."
        ),
    },
    {
        "key": "instruments",
        "prompt": (
            "What instruments are playing in this track? Describe each instrument's role, "
            "its tone/timbre, and how it contributes to the overall sound."
        ),
    },
]


class FlamingoWorker(StreamWorker):
    STREAM = "sap:flamingo:pending"
    GROUP = "flamingo-workers"

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self._model_loaded = False

    def _ensure_model(self):
        """Lazy-load the Music Flamingo model on first use."""
        if self._model_loaded:
            return

        logger.info("Loading Music Flamingo model: %s", MODEL_ID)
        from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto"
        )
        self._model_loaded = True
        logger.info("Music Flamingo model loaded successfully")

    async def process(self, message_id: str, data: dict) -> None:
        session_id = data["session_id"]
        minio_key = data["minio_key"]
        sr = int(data.get("sample_rate", settings.sample_rate))

        logger.info("Flamingo analysis starting for session %s", session_id)

        # Download PCM from MinIO
        pcm_bytes = await download_bytes(minio_key)
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)

        # Write to temp WAV file (Flamingo expects audio files, not raw PCM)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr, subtype="FLOAT")
            audio_path = tmp.name

        try:
            loop = asyncio.get_event_loop()

            # Ensure model is loaded (blocking, first call only)
            await loop.run_in_executor(None, self._ensure_model)

            # Run each analysis prompt
            sid = uuid.UUID(session_id)
            results = {}

            for prompt_config in ANALYSIS_PROMPTS:
                key = prompt_config["key"]
                prompt_text = prompt_config["prompt"]

                logger.info("Running Flamingo prompt '%s' for session %s", key, session_id)

                response = await loop.run_in_executor(
                    None, self._run_inference, audio_path, prompt_text,
                )

                results[key] = response

                # Store each analysis result
                async with async_session_factory() as db:
                    db.add(FlamingoAnalysis(
                        session_id=sid,
                        prompt_key=key,
                        prompt_text=prompt_text,
                        response=response,
                        structured_data={},
                    ))
                    await db.commit()

        finally:
            os.unlink(audio_path)

        logger.info(
            "Flamingo analysis complete for session %s: %d prompts",
            session_id, len(results),
        )

    def _run_inference(self, audio_path: str, prompt_text: str) -> str:
        """Run a single inference with Music Flamingo."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)
        inputs["input_features"] = inputs["input_features"].to(self.model.dtype)

        outputs = self.model.generate(**inputs, max_new_tokens=500)

        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return decoded[0] if decoded else ""


async def main():
    logging.basicConfig(level=logging.INFO)
    worker = FlamingoWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
