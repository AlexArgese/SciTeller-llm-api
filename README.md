# SciTeller VM API (GPU Service)

## Overview
FastAPI service providing the **split + storytelling** pipeline on GPU.  
It loads two scripts:
- `splitting/infer_splitter.py`
- `storytelling/infer_from_splits.py`

Adapters (LoRA/checkpoints) are mounted as **read-only volumes** at runtime.

## Endpoints
- `GET /health` → `{"ok": true, "stage": "split+story"}`

(If you expose more endpoints, list them here.)

## Build
```bash
docker build -t sciteller-api:gpu .
```

## Run (example)
- Host paths (replace with your real ones):
  - `$HF_CACHE` → HF cache on host (reused between runs)
  - `$SPLIT_DIR` → splitter checkpoint dir (e.g. `/docker/.../out-splitter-qwen7b`)
  - `$STORY_DIR` → storyteller adapter dir (e.g. `/docker/.../qwen32b_storyteller_lora`)

```bash
docker run -d --name sciteller-api   --gpus all   -p 8018:8018   -e API_KEY=""   -e HF_HOME=/hf   -e HUGGINGFACE_HUB_CACHE=/hf/hub   -e TRANSFORMERS_CACHE=/hf/transformers   -e SPLITTER_SCRIPT="/app/splitting/infer_splitter.py"   -e STORYTELLER_SCRIPT="/app/storytelling/infer_from_splits.py"   -e SPLITTER_ADAPTER_PATH="/models/splitter/checkpoint-100"   -e STORYTELLER_ADAPTER="/models/story/final_best"   -v "$HF_CACHE:/hf"   -v "$SPLIT_DIR:/models/splitter:ro"   -v "$STORY_DIR:/models/story:ro"   sciteller-api:gpu   uvicorn two_stage_app:app --host 0.0.0.0 --port 8018 --workers 1
```

Health check:
```bash
curl http://localhost:8018/health
# -> {"ok":true,"stage":"split+story"}
```

## Environment Variables
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` → model/cache directories inside the container
- `SPLITTER_SCRIPT`, `STORYTELLER_SCRIPT` → script entrypoints
- `SPLITTER_ADAPTER_PATH`, `STORYTELLER_ADAPTER` → **paths inside the container** (mounted with `-v`)
- `API_KEY` → optional auth (if the backend expects it)

## Dockerfile Notes
- Base: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- Installs PyTorch CUDA 12.1 wheels and python deps:
  - `fastapi`, `uvicorn`, `transformers`, `peft`, `datasets`,
  `sentence-transformers`, `scikit-learn`, `bitsandbytes`, `accelerate`, `pynvml`
- Exposes port **8018**

## What’s NOT in Git
- **No model weights**: they’re mounted at runtime (`/models/...`) or pulled into `/hf`.
- `.dockerignore` ensures only required code is sent in the build context.

## Troubleshooting
- Make sure `--gpus all` and NVIDIA runtime are available on the host.
- Verify the mounted paths exist and contain the expected checkpoint folders.
- If first run is slow, it’s likely downloading model weights into `HF_HOME`.

## Security
- Don’t bake secrets into the image. Use environment variables at runtime.
