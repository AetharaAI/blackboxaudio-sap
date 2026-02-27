from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# --- Session ---

class SessionCreate(BaseModel):
    tenant_id: str = "default"
    user_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class SessionResponse(BaseModel):
    id: uuid.UUID
    tenant_id: str
    user_id: str | None
    status: str
    filename: str | None
    mime_type: str | None
    duration_sec: float | None
    sample_rate: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    ws_url: str | None = None

    # Frontend-compatible aliases
    @property
    def name(self) -> str | None:
        return self.filename

    @property
    def duration(self) -> float | None:
        return self.duration_sec

    model_config = {"from_attributes": True}

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        d["name"] = self.filename
        d["duration"] = self.duration_sec
        return d


class SessionStatusUpdate(BaseModel):
    status: str
    error_message: str | None = None


# --- Transcript ---

class TranscriptSegmentSchema(BaseModel):
    t_start: float
    t_end: float
    text: str
    confidence: float | None = None
    speaker: str | None = None
    is_final: bool = False
    word_timestamps: list[WordTimestamp] | None = None

    model_config = {"from_attributes": True}


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float | None = None


# --- Music ---

class ChordSegmentSchema(BaseModel):
    t_start: float
    t_end: float
    label: str
    confidence: float | None = None

    model_config = {"from_attributes": True}


class MusicAnalysisSchema(BaseModel):
    tempo_bpm: float | None = None
    tempo_confidence: float | None = None
    key_label: str | None = None
    key_scale: str | None = None
    key_confidence: float | None = None
    beat_times: list[float] = Field(default_factory=list)
    downbeat_times: list[float] = Field(default_factory=list)
    time_signature: str = "4/4"

    model_config = {"from_attributes": True}


# --- Perception Frame ---

class AudioFeatures(BaseModel):
    rms: float = 0.0
    spectral_centroid: float = 0.0


class MusicFrame(BaseModel):
    chord: str = "N"
    key: str = ""
    bpm: float = 0.0
    beat: bool = False


class SpeechFrame(BaseModel):
    text_partial: str = ""
    words: list[str] = Field(default_factory=list)


class PerceptionFrameSchema(BaseModel):
    session_id: str
    t: float
    audio: AudioFeatures = Field(default_factory=AudioFeatures)
    music: MusicFrame = Field(default_factory=MusicFrame)
    speech: SpeechFrame = Field(default_factory=SpeechFrame)


# --- Result ---

class SessionResult(BaseModel):
    session: SessionResponse
    music_analysis: MusicAnalysisSchema | None = None
    chords: list[ChordSegmentSchema] = Field(default_factory=list)
    transcript: list[TranscriptSegmentSchema] = Field(default_factory=list)
    frames: list[PerceptionFrameSchema] = Field(default_factory=list)


# --- WebSocket Messages ---

class WSMessage(BaseModel):
    type: str
    session_id: str
    data: dict | None = None
    is_final: bool = False
    error: str | None = None
