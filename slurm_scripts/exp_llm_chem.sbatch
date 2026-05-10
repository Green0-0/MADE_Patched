#!/bin/bash
#SBATCH --job-name=exp_llm_chem_made
#SBATCH --output=llm-chem-%A_%a.out
#SBATCH --error=llm-chem-%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=1-10

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ENDPOINT_FILE="${PWD}/.vllm_endpoint"
HEARTBEAT_FILE="${PWD}/.vllm_heartbeat"

echo "Waiting for vLLM endpoint file at ${ENDPOINT_FILE}..."
while [ ! -f "$ENDPOINT_FILE" ]; do
  sleep 10
done

export OPENAI_BASE_URL=$(cat "$ENDPOINT_FILE")
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
echo "Discovered vLLM endpoint: $OPENAI_BASE_URL"

# Verify endpoint is accessible before starting work
python - <<'PY'
import json
import time
import urllib.request
import os

url = os.environ.get("OPENAI_BASE_URL") + "/models"
deadline = time.time() + 60

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print("Successfully connected to the shared vLLM server.")
                raise SystemExit(0)
    except Exception:
        time.sleep(2)

raise SystemExit(f"Could not connect to vLLM endpoint at {url}")
PY

if [ $? -ne 0 ]; then
  exit 1
fi

# Keep heartbeat updated while this job runs so the vLLM server stays alive
touch "$HEARTBEAT_FILE"
while true; do
  sleep 300
  touch "$HEARTBEAT_FILE"
done &
HEARTBEAT_PID=$!

cleanup() {
  kill $HEARTBEAT_PID 2>/dev/null || true
}
trap cleanup EXIT

uv run scripts/run_baseline_experiments.py \
  --agent-configs chemeleon_llm_planner \
  --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json \
  --max-systems 10 \
  --system-index $((SLURM_ARRAY_TASK_ID - 1)) \
  --budget 50 \
  --num-episodes 5 \
  --use-wandb
