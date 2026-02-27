FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-dev libsamplerate0 libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/music_worker /app/services/music_worker
RUN pip install --no-cache-dir --pre /app/services/music_worker

CMD ["python", "-m", "music_worker.worker"]
