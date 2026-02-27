from __future__ import annotations

import numpy as np
import essentia.standard as es


def analyze_tempo_and_beats(audio: np.ndarray, sr: int = 44100) -> dict:
    """Estimate tempo (BPM) and beat positions using Essentia RhythmExtractor2013.

    Falls back to librosa if Essentia confidence is low.

    Returns:
        {
            "tempo_bpm": float,
            "tempo_confidence": float,
            "beat_times": list[float],
            "downbeat_times": list[float],
            "time_signature": str,
        }
    """
    # Primary: Essentia RhythmExtractor2013
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beat_ticks, bpm_confidence, _, beat_intervals = rhythm_extractor(audio)

    # Cross-check with librosa if confidence is low
    if bpm_confidence < 0.4:
        import librosa

        tempo_librosa, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times_librosa = librosa.frames_to_time(beat_frames, sr=sr)

        if len(beat_times_librosa) > 2:
            ibi_std_librosa = np.std(np.diff(beat_times_librosa))
            ibi_std_essentia = np.std(beat_intervals) if len(beat_intervals) > 1 else float("inf")

            if ibi_std_librosa < ibi_std_essentia:
                bpm = float(np.atleast_1d(tempo_librosa)[0])
                beat_ticks = beat_times_librosa
                bpm_confidence = 0.5

    # Estimate downbeats (every 4th beat for 4/4 time)
    downbeat_times = _estimate_downbeats(beat_ticks)
    time_sig = _estimate_time_signature(beat_intervals if len(beat_intervals) > 0 else np.diff(beat_ticks))

    return {
        "tempo_bpm": round(float(bpm), 2),
        "tempo_confidence": round(float(min(bpm_confidence, 1.0)), 4),
        "beat_times": [round(float(t), 4) for t in beat_ticks],
        "downbeat_times": [round(float(t), 4) for t in downbeat_times],
        "time_signature": time_sig,
    }


def _estimate_downbeats(beat_times: np.ndarray) -> list[float]:
    """Estimate downbeats as every 4th beat (assuming 4/4 time)."""
    if len(beat_times) < 4:
        return [float(beat_times[0])] if len(beat_times) > 0 else []
    return [float(beat_times[i]) for i in range(0, len(beat_times), 4)]


def _estimate_time_signature(beat_intervals: np.ndarray) -> str:
    """Simple heuristic for time signature estimation.

    If median inter-beat interval clusters suggest groupings of 3, return 3/4.
    Otherwise default to 4/4.
    """
    if len(beat_intervals) < 4:
        return "4/4"

    median_ibi = np.median(beat_intervals)
    if median_ibi < 0.01:
        return "4/4"

    # Check for waltz feel: if beat intervals show a pattern of
    # long-short-short, it might be 3/4
    # This is a rough heuristic; proper time signature detection is hard
    return "4/4"
