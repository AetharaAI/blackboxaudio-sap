from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

CHATTERBOX_BASE_URL = os.environ.get("CHATTERBOX_URL", "https://tts.aetherpro.us")


class ChatterboxClient:
    """Client for the Chatterbox TTS Server API.

    Supports both predefined voices and voice cloning via reference audio.
    Talks to the already-running Chatterbox instance rather than loading
    any model locally — no GPU needed in this worker.
    """

    def __init__(self, base_url: str = CHATTERBOX_BASE_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_predefined_voices(self) -> list[str]:
        """List available predefined voices."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/get_predefined_voices")
            resp.raise_for_status()
            return resp.json()

    async def get_model_info(self) -> dict:
        """Get model information from the server."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/model-info")
            resp.raise_for_status()
            return resp.json()

    async def synthesize(
        self,
        text: str,
        voice_mode: str = "predefined",
        predefined_voice_id: str | None = None,
        reference_audio_filename: str | None = None,
        output_format: str = "wav",
        split_text: bool = True,
        chunk_size: int = 120,
        temperature: float | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        seed: int | None = None,
        speed_factor: float | None = None,
    ) -> bytes:
        """Synthesize text using the Chatterbox /tts endpoint.

        Args:
            text: Text to synthesize.
            voice_mode: "predefined" or "clone".
            predefined_voice_id: Filename of predefined voice (when voice_mode="predefined").
            reference_audio_filename: Reference audio filename (when voice_mode="clone").
            output_format: "wav", "opus", or "mp3".
            split_text: Whether to split long text into chunks.
            chunk_size: Max characters per chunk (50-500).
            temperature: Generation temperature (controls variation).
            exaggeration: Voice exaggeration factor.
            cfg_weight: Classifier-free guidance weight.
            seed: Random seed for reproducibility.
            speed_factor: Playback speed multiplier.

        Returns:
            Audio bytes in the requested format.
        """
        payload = {
            "text": text,
            "voice_mode": voice_mode,
            "output_format": output_format,
            "split_text": split_text,
            "chunk_size": chunk_size,
        }

        if predefined_voice_id:
            payload["predefined_voice_id"] = predefined_voice_id
        if reference_audio_filename:
            payload["reference_audio_filename"] = reference_audio_filename
        if temperature is not None:
            payload["temperature"] = temperature
        if exaggeration is not None:
            payload["exaggeration"] = exaggeration
        if cfg_weight is not None:
            payload["cfg_weight"] = cfg_weight
        if seed is not None:
            payload["seed"] = seed
        if speed_factor is not None:
            payload["speed_factor"] = speed_factor

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/tts", json=payload)
            resp.raise_for_status()
            return resp.content

    async def synthesize_openai_compat(
        self,
        text: str,
        voice: str,
        model: str = "chatterbox",
        response_format: str = "wav",
        speed: float = 1.0,
        seed: int | None = None,
    ) -> bytes:
        """Synthesize using the OpenAI-compatible /v1/audio/speech endpoint.

        Simpler interface — good for quick generation with a named voice.
        """
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        if seed is not None:
            payload["seed"] = seed

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/audio/speech", json=payload
            )
            resp.raise_for_status()
            return resp.content

    async def upload_reference_audio(
        self, filepath: str, filename: str | None = None
    ) -> dict:
        """Upload a reference audio file for voice cloning."""
        fname = filename or os.path.basename(filepath)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(filepath, "rb") as f:
                resp = await client.post(
                    f"{self.base_url}/upload_reference",
                    files={"file": (fname, f, "audio/wav")},
                )
            resp.raise_for_status()
            return resp.json()

    async def upload_reference_bytes(
        self, data: bytes, filename: str = "reference.wav"
    ) -> dict:
        """Upload reference audio bytes for voice cloning."""
        import io

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/upload_reference",
                files={"file": (filename, io.BytesIO(data), "audio/wav")},
            )
            resp.raise_for_status()
            return resp.json()
