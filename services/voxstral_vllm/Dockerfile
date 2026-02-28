FROM vllm/vllm-openai:nightly

RUN pip install --no-cache-dir "mistral-common[soundfile]" soundfile libsndfile1 || true
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "mistral-common[soundfile]"
