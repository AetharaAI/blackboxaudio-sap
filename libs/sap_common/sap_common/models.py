from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class SessionStatus(str, enum.Enum):
    created = "created"
    uploading = "uploading"
    preprocessing = "preprocessing"
    analyzing = "analyzing"
    aligning = "aligning"
    completed = "completed"
    completed_partial = "completed_partial"
    failed = "failed"


class AudioSession(Base):
    __tablename__ = "audio_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status"),
        nullable=False,
        default=SessionStatus.created,
    )
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[int] = mapped_column(Integer, default=44100)
    minio_raw_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    minio_pcm_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    transcript_segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    chord_segments: Mapped[list[MusicChordSegment]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    music_analysis: Mapped[MusicAnalysis | None] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )
    perception_frames: Mapped[list[PerceptionFrame]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    flamingo_analyses: Mapped[list[FlamingoAnalysis]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_sessions_tenant", "tenant_id", "created_at"),
        Index("idx_sessions_status", "status"),
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_sessions.id", ondelete="CASCADE"), nullable=False
    )
    t_start: Mapped[float] = mapped_column(Float, nullable=False)
    t_end: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    speaker: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="whisper")
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    word_timestamps: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped[AudioSession] = relationship(back_populates="transcript_segments")

    __table_args__ = (
        Index("idx_transcript_session", "session_id", "t_start"),
        Index("idx_transcript_final", "session_id", "is_final"),
    )


class MusicChordSegment(Base):
    __tablename__ = "music_chord_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_sessions.id", ondelete="CASCADE"), nullable=False
    )
    t_start: Mapped[float] = mapped_column(Float, nullable=False)
    t_end: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped[AudioSession] = relationship(back_populates="chord_segments")

    __table_args__ = (
        Index("idx_chords_session", "session_id", "t_start"),
    )


class MusicAnalysis(Base):
    __tablename__ = "music_analysis"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audio_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    tempo_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    key_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    key_scale: Mapped[str | None] = mapped_column(String(8), nullable=True)
    key_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    beat_times: Mapped[list[float]] = mapped_column(
        ARRAY(Float), nullable=False, default=list
    )
    downbeat_times: Mapped[list[float] | None] = mapped_column(
        ARRAY(Float), nullable=True, default=list
    )
    time_signature: Mapped[str] = mapped_column(String(8), default="4/4")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped[AudioSession] = relationship(back_populates="music_analysis")


class PerceptionFrame(Base):
    __tablename__ = "perception_frames"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_sessions.id", ondelete="CASCADE"), nullable=False
    )
    t: Mapped[float] = mapped_column(Float, nullable=False)
    frame_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped[AudioSession] = relationship(back_populates="perception_frames")

    __table_args__ = (
        Index("idx_frames_session", "session_id", "t"),
    )


class FlamingoAnalysis(Base):
    __tablename__ = "flamingo_analysis"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_sessions.id", ondelete="CASCADE"), nullable=False
    )
    prompt_key: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    structured_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped[AudioSession] = relationship(back_populates="flamingo_analyses")

    __table_args__ = (
        Index("idx_flamingo_session", "session_id", "prompt_key"),
    )
