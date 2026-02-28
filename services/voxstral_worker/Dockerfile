FROM python:3.11-slim

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/voxstral_worker /app/services/voxstral_worker
RUN pip install --no-cache-dir /app/services/voxstral_worker

CMD ["python", "-m", "voxstral_worker.worker"]
