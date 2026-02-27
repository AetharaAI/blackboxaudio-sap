from __future__ import annotations

import bisect


FRAME_RESOLUTION = 0.250  # 250ms


def build_perception_frames(
    session_id: str,
    duration_sec: float,
    audio_features: list[dict],
    tempo_result: dict,
    key_result: dict,
    chord_segments: list[dict],
    transcript_segments: list[dict],
) -> list[dict]:
    """Align all analysis results onto a 250ms grid.

    Strategy:
    - Audio features: already at 250ms resolution, direct map.
    - Beats: binary flag per frame (beat within this 250ms window?).
    - Chords: lookup which chord segment contains each frame's center time.
    - Key/BPM: constant across all frames (global properties).
    - Speech: find transcript words overlapping this frame.
    """
    beat_times = sorted(tempo_result.get("beat_times", []))
    chord_index = sorted(chord_segments, key=lambda c: c["t_start"])

    # Build word list from transcript
    all_words = []
    for seg in transcript_segments:
        for w in seg.get("words", []):
            if isinstance(w, dict):
                all_words.append(w)
    all_words.sort(key=lambda w: w.get("start", 0))

    # Index audio features by time
    feature_map = {round(f["t"], 4): f for f in audio_features}

    # Key/BPM are global
    key_str = f"{key_result.get('key_label', '')}:{key_result.get('key_scale', '')[:3]}"
    bpm = tempo_result.get("tempo_bpm", 0.0)

    num_frames = int(duration_sec / FRAME_RESOLUTION) + 1
    frames = []

    for i in range(num_frames):
        t = round(i * FRAME_RESOLUTION, 4)
        t_end = t + FRAME_RESOLUTION

        # Audio features
        af = feature_map.get(t, {"rms": 0.0, "spectral_centroid": 0.0})

        # Beat detection: is there a beat within [t, t + 0.250)?
        beat_idx = bisect.bisect_left(beat_times, t)
        has_beat = beat_idx < len(beat_times) and beat_times[beat_idx] < t_end

        # Chord lookup
        chord_label = "N"
        for cs in chord_index:
            if cs["t_start"] <= t < cs["t_end"]:
                chord_label = cs["label"]
                break
            if cs["t_start"] > t:
                break

        # Speech: collect words overlapping this frame
        frame_words = []
        for w in all_words:
            w_end = w.get("end", 0)
            w_start = w.get("start", 0)
            if w_end < t:
                continue
            if w_start >= t_end:
                break
            frame_words.append(w.get("word", ""))
        frame_text = " ".join(frame_words).strip()

        frames.append({
            "session_id": session_id,
            "t": t,
            "audio": {
                "rms": af.get("rms", 0.0),
                "spectral_centroid": af.get("spectral_centroid", 0.0),
            },
            "music": {
                "chord": chord_label,
                "key": key_str,
                "bpm": bpm,
                "beat": has_beat,
            },
            "speech": {
                "text_partial": frame_text,
                "words": frame_words,
            },
        })

    return frames
