FROM python:3.11-slim

WORKDIR /app

COPY libs/sap_common /app/libs/sap_common
RUN pip install --no-cache-dir /app/libs/sap_common

COPY services/aligner /app/services/aligner
RUN pip install --no-cache-dir /app/services/aligner

CMD ["python", "-m", "aligner.worker"]
