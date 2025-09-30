FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY two_stage_app.py /app/
COPY splitting/infer_splitter.py /app/splitting/infer_splitter.py
COPY storytelling/infer_from_splits.py /app/storytelling/infer_from_splits.py

# 1) PyTorch CUDA 12.1 from repo PyTorch
RUN pip3 install --no-cache-dir \
    "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "torchaudio==2.3.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# 2) PyPI
RUN pip3 install --no-cache-dir \
    fastapi==0.111.0 uvicorn[standard]==0.30.0 \
    transformers==4.43.3 peft==0.11.1 datasets==2.20.0 \
    sentence-transformers==2.7.0 scikit-learn==1.5.1 \
    bitsandbytes==0.43.3 accelerate==0.33.0 pynvml==11.5.0


ENV HF_HOME=/hf-cache
ENV SPLITTER_SCRIPT=/app/splitting/infer_splitter.py
ENV STORYTELLER_SCRIPT=/app/storytelling/infer_from_splits.py
ENV SPLITTER_ADAPTER_PATH=/models/out-splitter-qwen7b/checkpoint-100
ENV STORYTELLER_ADAPTER=/models/qwen32b_storyteller_lora/final_best
ENV API_KEY=""

EXPOSE 8018
CMD ["uvicorn", "two_stage_app:app", "--host", "0.0.0.0", "--port", "8018", "--workers", "1"]
