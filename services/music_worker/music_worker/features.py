from __future__ import annotations

import numpy as np
import librosa


def compute_frame_features(
    audio: np.ndarray,
    sr: int = 44100,
    frame_duration: float = 0.250,
) -> list[dict]:
    """Compute per-frame audio features at 250ms resolution.

    Returns list of {"t": float, "rms": float, "spectral_centroid": float}.
    """
    hop_length = int(sr * frame_duration)  # 11025 samples per 250ms
    frame_length = hop_length

    # RMS energy
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=max(frame_length, 2048), hop_length=hop_length
    )[0]

    num_frames = min(len(rms), len(centroid))
    results = []
    for i in range(num_frames):
        results.append({
            "t": round(i * frame_duration, 4),
            "rms": round(float(rms[i]), 6),
            "spectral_centroid": round(float(centroid[i]), 2),
        })

    return results
