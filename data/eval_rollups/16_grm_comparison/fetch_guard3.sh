#!/bin/bash
set -uo pipefail
export HF_HUB_OFFLINE=0
export HF_TOKEN=$(cat "$HF_HOME/token")
${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("meta-llama/Llama-Guard-3-8B",
                      allow_patterns=["*.json","*.safetensors","tokenizer*","*.model"],
                      max_workers=8)
print("GUARD3PATH", p, flush=True)
PY
echo "GUARD3DONE $(date -Is)"
