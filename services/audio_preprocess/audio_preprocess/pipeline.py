from __future__ import annotations

import subprocess
import tempfile

import numpy as np
from scipy.signal import butter, sosfilt


def decode_to_pcm(input_path: str, target_sr: int = 44100) -> np.ndarray:
    """Decode any audio format to float32 mono PCM using ffmpeg.

    Pipes raw float32 little-endian PCM from ffmpeg stdout.
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-v", "quiet",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    return np.frombuffer(result.stdout, dtype=np.float32)


def get_duration(input_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
    return float(result.stdout.strip())


def highpass_filter(
    audio: np.ndarray, sr: int = 44100, cutoff: float = 60.0
) -> np.ndarray:
    """4th-order Butterworth high-pass at cutoff Hz."""
    sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """RMS normalization to consistent loudness."""
    current_rms = np.sqrt(np.mean(audio**2))
    if current_rms < 1e-8:
        return audio
    gain = target_rms / current_rms
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def preprocess(input_path: str, sr: int = 44100) -> tuple[np.ndarray, float]:
    """Full preprocessing pipeline: decode -> highpass -> normalize.

    Returns (audio_array, duration_sec).
    """
    duration = get_duration(input_path)
    audio = decode_to_pcm(input_path, sr)
    audio = highpass_filter(audio, sr, cutoff=60.0)
    audio = normalize_rms(audio, target_rms=0.1)
    return audio, duration
