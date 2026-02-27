FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/audio_gateway /app/services/audio_gateway
RUN pip install --no-cache-dir /app/services/audio_gateway

EXPOSE 8000
CMD ["uvicorn", "audio_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
