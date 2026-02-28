FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libsndfile1 \
    git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools

# Install Music Flamingo's custom transformers fork
RUN pip install --no-cache-dir "git+https://github.com/lashahub/transformers@modular-mf" accelerate

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/flamingo_worker /app/services/flamingo_worker
RUN pip install --no-cache-dir /app/services/flamingo_worker

# Model cache directory
ENV HF_HOME=/models
ENV CUDA_VISIBLE_DEVICES=0
VOLUME /models

CMD ["python", "-m", "flamingo_worker.worker"]
