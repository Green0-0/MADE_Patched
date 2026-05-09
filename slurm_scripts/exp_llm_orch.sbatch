#!/bin/bash
#SBATCH --job-name=exp_llm_orch_made
#SBATCH --output=llm-orch-%A_%a.out
#SBATCH --error=llm-orch-%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --array=1-10

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
VLLM_PYTHON="${VLLM_VENV_PATH}/bin/python"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "vLLM venv not found: ${VLLM_VENV_PATH}" >&2
  exit 1
fi

"${VLLM_PYTHON}" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("vllm")
except Exception as exc:
    print(f"vllm import failed in vLLM venv: {exc}", file=sys.stderr)
    raise
PY

export MODEL_NAME="QuantTrio/Qwen3.5-9B-AWQ"
export VLLM_HOST="127.0.0.1"
export VLLM_PORT=$(( 31000 + SLURM_ARRAY_TASK_ID ))
export MASTER_PORT=$(( 32000 + SLURM_ARRAY_TASK_ID ))
export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

cleanup_vllm() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT

"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${MODEL_NAME}" \
  --gpu-memory-utilization 0.75 \
  --max-model-len 49152 &
VLLM_PID=$!

python - <<'PY'
import json
import time
import urllib.request
import os

# Grab the port we dynamically assigned in bash
port = os.environ.get("VLLM_PORT")
url = f"http://127.0.0.1:{port}/v1/models"

deadline = time.time() + 600
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("data"):
                    print(f"vLLM ready on port {port}")
                    raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)

raise SystemExit(f"vLLM did not become ready on port {port}: {last_error}")
PY

if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "vLLM process exited before baseline run" >&2
  exit 1
fi

uv run scripts/run_baseline_experiments.py \
  --agent-configs llm_react_orchestrator \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --system-index $((SLURM_ARRAY_TASK_ID - 1)) \
  --budget 50 \
  --num-episodes 5 \
  --use-wandb