from __future__ import annotations

import numpy as np
import essentia.standard as es


def analyze_chords(audio: np.ndarray, sr: int = 44100) -> list[dict]:
    """Frame-by-frame chord recognition using Essentia's ChordsDetection.

    Pipeline:
    1. Windowing -> Spectrum -> SpectralPeaks -> HPCP (36 bins)
    2. ChordsDetection on HPCP sequence
    3. Post-process: merge consecutive identical chords, filter <100ms

    Returns:
        List of {"t_start": float, "t_end": float, "label": str, "confidence": float}
    """
    FRAME_SIZE = 8192   # ~186ms at 44100 Hz — good spectral resolution for chords
    HOP_SIZE = 2048     # ~46ms hop

    # Build processing chain
    w = es.Windowing(type="blackmanharris92")
    spectrum = es.Spectrum(size=FRAME_SIZE)
    spectral_peaks = es.SpectralPeaks(
        sampleRate=sr,
        magnitudeThreshold=1e-05,
        maxPeaks=60,
        orderBy="magnitude",
    )
    hpcp_algo = es.HPCP(
        sampleRate=sr,
        size=36,
        referenceFrequency=440.0,
        harmonics=8,
        bandPreset=True,
        minFrequency=40.0,
        maxFrequency=5000.0,
        weightType="cosine",
        nonLinear=False,
    )

    # Compute HPCP for each frame
    num_frames = max(0, 1 + (len(audio) - FRAME_SIZE) // HOP_SIZE)
    hpcp_frames = []

    for i in range(num_frames):
        start = i * HOP_SIZE
        frame = audio[start : start + FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
        windowed = w(frame.astype(np.float32))
        spec = spectrum(windowed)
        freqs, mags = spectral_peaks(spec)
        h = hpcp_algo(freqs, mags)
        hpcp_frames.append(h)

    if not hpcp_frames:
        return []

    hpcp_array = np.array(hpcp_frames)

    # Run chord detection
    chords_detection = es.ChordsDetection(
        hopSize=HOP_SIZE,
        sampleRate=sr,
        windowSize=2.0,
    )
    chords, strengths = chords_detection(hpcp_array)

    # Merge consecutive identical chords
    frame_duration = HOP_SIZE / sr
    segments = []
    current_chord = None
    current_start = 0.0
    strength_sum = 0.0
    count = 0

    for i, (chord, strength) in enumerate(zip(chords, strengths)):
        t = i * frame_duration
        if chord != current_chord:
            if current_chord is not None and current_chord != "N":
                segments.append({
                    "t_start": round(current_start, 4),
                    "t_end": round(t, 4),
                    "label": current_chord,
                    "confidence": round(strength_sum / max(count, 1), 4),
                })
            current_chord = chord
            current_start = t
            strength_sum = float(strength)
            count = 1
        else:
            strength_sum += float(strength)
            count += 1

    # Flush last segment
    if current_chord is not None and current_chord != "N":
        total_duration = len(audio) / sr
        segments.append({
            "t_start": round(current_start, 4),
            "t_end": round(total_duration, 4),
            "label": current_chord,
            "confidence": round(strength_sum / max(count, 1), 4),
        })

    # Filter segments shorter than 100ms (likely spurious)
    segments = [s for s in segments if (s["t_end"] - s["t_start"]) >= 0.1]

    return segments
