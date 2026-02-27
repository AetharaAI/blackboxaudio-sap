FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/audio_preprocess /app/services/audio_preprocess
RUN pip install --no-cache-dir /app/services/audio_preprocess

CMD ["python", "-m", "audio_preprocess.worker"]
