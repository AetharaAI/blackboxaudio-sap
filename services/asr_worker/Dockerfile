FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/asr_worker /app/services/asr_worker
RUN pip install --no-cache-dir /app/services/asr_worker

# Model cache directory
ENV HF_HOME=/models
VOLUME /models

CMD ["python", "-m", "asr_worker.worker"]
