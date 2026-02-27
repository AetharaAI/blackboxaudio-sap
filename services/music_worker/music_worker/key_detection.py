from __future__ import annotations

import numpy as np
import essentia.standard as es


def analyze_key(audio: np.ndarray, sr: int = 44100) -> dict:
    """Estimate musical key using Essentia's KeyExtractor.

    Uses the 'temperley' profile which handles pop/rock well.

    Returns:
        {
            "key_label": str,        # e.g. "G", "Bb", "F#"
            "key_scale": str,        # "major" or "minor"
            "key_confidence": float, # 0..1
        }
    """
    key_extractor = es.KeyExtractor(profileType="temperley", sampleRate=sr)
    key, scale, confidence = key_extractor(audio)

    return {
        "key_label": key,
        "key_scale": scale,
        "key_confidence": round(float(confidence), 4),
    }
