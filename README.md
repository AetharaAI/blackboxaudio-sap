# Structured Acoustic Perception (SAP)

Audio perception engine that converts raw audio into time-aligned structured signals: speech transcription, music analysis (tempo/key/chords/beats), and fused Perception Frames at 250ms resolution.

## Quick Start

**Local dev (any GPU):**
```bash
cd sap/
docker compose up --build
```

**Production on L4-360 VM (pins ASR to GPU 3, mounts model cache):**
```bash
cd sap/
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

Gateway available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

MinIO console at `http://localhost:9101` (credentials in `.env`).

## Prerequisites

- Docker Engine 24+ with Compose v2
- NVIDIA Container Toolkit (for ASR worker GPU access)
- ~15 GB disk for Docker images + model cache
- NVIDIA GPU with 6+ GB VRAM (see Model Requirements below)

### Install NVIDIA Container Toolkit (if not already)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Deployment

### 1. Environment

The defaults in `docker-compose.yml` work out of the box for local dev. For production, copy and customize:

```bash
cp .env.example .env
# Edit .env with production values (Postgres password, MinIO credentials, etc.)
```

### 2. Pre-download Whisper Models (Optional)

First run will auto-download models, but you can pre-cache them:

```bash
pip install faster-whisper
python scripts/download_models.py
```

If models are already cached in `~/.cache/huggingface/hub/`, mount that into the container (see GPU Deployment below).

### 3. Build and Run

```bash
# Full stack (requires GPU)
docker compose up --build

# Infrastructure only (no GPU needed)
docker compose up postgres valkey minio

# Non-GPU services only
docker compose up postgres valkey minio audio-gateway audio-preprocess music-worker aligner tts-worker

# Add ASR worker separately (requires GPU)
docker compose up asr-worker
```

### 4. Verify

```bash
# Health check
curl http://localhost:8000/health/ready

# Create a session and upload
curl -X POST http://localhost:8000/v1/audio/sessions \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "test"}'

# Upload audio (replace SESSION_ID and file path)
curl -X POST http://localhost:8000/v1/audio/sessions/{SESSION_ID}/upload \
  -F "file=@/path/to/song.mp3"
```

## Model Requirements

### ASR Models (GPU)

SAP uses dual-pass Whisper for transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper):

| Model | HuggingFace ID | VRAM (float16) | Purpose |
|-------|---------------|----------------|---------|
| **distil-large-v3** | `Systran/faster-whisper-distil-large-v3` | ~1.5 GB | Streaming pass (low latency, partial transcripts) |
| **large-v3** | `Systran/faster-whisper-large-v3` | ~3.1 GB | Final pass (high accuracy, word timestamps) |
| **Combined** | — | **~4.5 GB** | Both models loaded simultaneously |

### Music Analysis (CPU only)

No GPU needed. Uses:
- **Essentia** — tempo/beats (RhythmExtractor2013), key (KeyExtractor), chords (HPCP + ChordsDetection)
- **librosa** — spectral features (RMS, centroid), beat fallback

### TTS (External)

No local model. Proxies to Chatterbox TTS Server at `tts.aetherpro.us`. Change via `CHATTERBOX_URL` env var.

## GPU Deployment Guide

### Single GPU (Minimum)

One NVIDIA GPU with **6+ GB VRAM** handles both Whisper models at float16.

Any of these work: L4 (24GB), T4 (16GB), A10 (24GB), RTX 3060+ (12GB+), RTX 4060+ (8GB+).

### L4-360 VM Layout (4x L4, 24GB each)

```
GPU 0:  9.2 / 23 GB  — text-embeddings (embed + reranker) + Chatterbox TTS
GPU 1: 19.5 / 23 GB  — VLLM Qwen-VL TP0 (tensor parallel, vision model)
GPU 2: 19.5 / 23 GB  — VLLM Qwen-VL TP1
GPU 3:    0 / 23 GB  — SAP ASR Worker ← 5.1 GB needed, 23 GB available
```

Existing containers on the VM:
- `chatterbox-tts-server:cu128` → port 8004 (nginx → tts.aetherpro.us)
- `vllm-openai` (redwatch-qwen-vl) → port 8008
- `text-embeddings-inference` x2 → ports 8080, 8081

SAP ports (no conflicts): 8000 (gateway), 5433 (postgres), 6385 (valkey), 9100/9101 (minio)

Use `docker-compose.prod.yml` to pin ASR to GPU 3 and mount the model cache:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

This override pins `asr-worker` to `device_ids: ["3"]` and mounts `/home/ubuntu/.cache/huggingface:/models` so you don't re-download the ~3.1 GB `large-v3` model that's already cached.

You still need `distil-large-v3` (~1.5 GB) — it will auto-download on first boot, or pre-download:

```bash
python scripts/download_models.py --model-dir /home/ubuntu/.cache/huggingface
```

### VRAM Budget on L4 (24GB)

```
faster-whisper-large-v3     ~3.1 GB  (final pass)
faster-whisper-distil-large-v3  ~1.5 GB  (streaming pass)
CUDA overhead               ~0.5 GB
─────────────────────────────────────
Total SAP ASR               ~5.1 GB
Free on L4                  ~18.9 GB  (plenty of headroom)
```

### Lower VRAM Options

If deploying on a smaller GPU (4-6GB), switch to int8 quantization:

```yaml
  asr-worker:
    environment:
      COMPUTE_TYPE: int8   # ~50% less VRAM, slight accuracy trade-off
```

## Architecture

```
┌──────────────┐
│   Clients    │  Browser / API / Agents
└──────┬───────┘
       │ REST + WebSocket
       ▼
┌──────────────┐
│   Gateway    │  FastAPI · :8000
│  (sessions,  │  Upload, Results, TTS, WebSocket
│   routes)    │
└──────┬───────┘
       │ Valkey Streams
  ┌────┴────┐
  ▼         ▼
┌────────┐ ┌───────────┐
│Preproc │ │TTS Worker │→ Chatterbox API
│ ffmpeg │ └───────────┘
└───┬────┘
    │ fan-out
 ┌──┴──┐
 ▼     ▼
┌───┐ ┌─────┐
│ASR│ │Music│
│GPU│ │ CPU │
└─┬─┘ └──┬──┘
  └───┬───┘
      ▼
┌──────────┐
│ Aligner  │→ Perception Frames (250ms grid)
└──────────┘
      │
  ┌───┴───┐
  ▼       ▼
┌────┐  ┌─────┐
│ PG │  │MinIO│
└────┘  └─────┘
```

## Services

| Service | Port | GPU | Purpose |
|---------|------|-----|---------|
| `postgres` | 5433 | — | Session metadata, analysis results, frames |
| `valkey` | 6385 | — | Message bus (streams), worker coordination |
| `minio` | 9100/9101 | — | Audio file storage |
| `audio-gateway` | 8000 | — | REST API + WebSocket relay |
| `audio-preprocess` | — | — | ffmpeg decode, normalize, HPF |
| `asr-worker` | — | 1x GPU | Dual Whisper transcription |
| `music-worker` | — | — | Essentia + librosa analysis |
| `aligner` | — | — | Perception Frame fusion |
| `tts-worker` | — | — | Chatterbox TTS client |

## API Overview

### Audio Analysis
```
POST   /v1/audio/sessions                           Create session
POST   /v1/audio/sessions/{id}/upload               Upload audio
GET    /v1/audio/sessions/{id}/result               Full results
GET    /v1/audio/sessions/{id}/stream               Audio playback (Range)
GET    /v1/audio/sessions/{id}/frames?t_start=&t_end=  Paginated frames
WS     /ws/audio/{session_id}                       Live updates
```

### Exports
```
GET    /v1/audio/sessions/{id}/export/lyrics.txt    Lyrics
GET    /v1/audio/sessions/{id}/export/chords.json   Chords + key + tempo
GET    /v1/audio/sessions/{id}/export/beats.json    Beat grid
GET    /v1/audio/sessions/{id}/export/frames.jsonl  All frames (NDJSON)
```

### TTS
```
POST   /v1/audio/tts/synthesize                     Async (queued)
POST   /v1/audio/tts/generate                       Sync (immediate)
GET    /v1/audio/tts/{id}/audio                     Download result
GET    /v1/audio/tts/voices                         List voices
```

TTS powered by [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI (MIT License). All generated audio includes Perth watermarking.

## Troubleshooting

**ASR worker won't start (no GPU)**
```bash
# Run without ASR — everything else works, just no transcription
docker compose up postgres valkey minio audio-gateway audio-preprocess music-worker aligner tts-worker
```

**Models downloading slowly**
```bash
# Pre-download on the host, then mount cache
python scripts/download_models.py
# Add to docker-compose.yml asr-worker volumes:
#   - ~/.cache/huggingface:/models
```

**Port conflicts**
```bash
# Change in docker-compose.yml ports section
# e.g., "8080:8000" to use port 8080 for the gateway
```

**Health check failing**
```bash
curl http://localhost:8000/health/ready
# Shows which dependency is down (postgres, valkey, or minio)
```
