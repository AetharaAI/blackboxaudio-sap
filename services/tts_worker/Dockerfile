FROM python:3.11-slim

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/tts_worker /app/services/tts_worker
RUN pip install --no-cache-dir /app/services/tts_worker

CMD ["python", "-m", "tts_worker.worker"]
