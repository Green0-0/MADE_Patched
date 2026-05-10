#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --output=vllm-server-%j.out
#SBATCH --error=vllm-server-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4

# Paths for coordination
ENDPOINT_FILE="${PWD}/.vllm_endpoint"
HEARTBEAT_FILE="${PWD}/.vllm_heartbeat"

# Clean up any stale endpoint file
rm -f "$ENDPOINT_FILE"
# Initialize heartbeat so it doesn't instantly timeout
touch "$HEARTBEAT_FILE"

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

export MODEL_NAME="QuantTrio/Qwen3.5-9B-AWQ"
# Must bind to 0.0.0.0 so other nodes in the cluster can connect
export VLLM_HOST="0.0.0.0"

# Dynamically find a free port on this node to avoid conflicts
export VLLM_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Starting vLLM on port ${VLLM_PORT}..."

cleanup_vllm() {
  echo "Shutting down vLLM..."
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  rm -f "$ENDPOINT_FILE"
}
trap cleanup_vllm EXIT

# Start vLLM in the background
"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 49152 &
VLLM_PID=$!

# Wait for vLLM to become healthy before publishing the endpoint
python - <<'PY'
import json
import time
import urllib.request
import os

port = os.environ.get("VLLM_PORT")
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 600

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print(f"vLLM ready internally on port {port}")
                raise SystemExit(0)
    except Exception:
        time.sleep(2)

raise SystemExit(f"vLLM did not become ready.")
PY

if [ $? -ne 0 ]; then
  echo "vLLM failed to start."
  exit 1
fi

# Publish the endpoint for worker scripts to find
# We use hostname instead of 127.0.0.1 so workers on other nodes can resolve it
HOSTNAME=$(hostname)
echo "http://${HOSTNAME}:${VLLM_PORT}/v1" > "$ENDPOINT_FILE"
echo "Published endpoint: http://${HOSTNAME}:${VLLM_PORT}/v1"

# Heartbeat monitor loop
echo "Starting heartbeat monitor. Will shutdown after 1 hour of inactivity."
while true; do
  sleep 60
  
  # Ensure the vLLM process hasn't crashed
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM process exited unexpectedly."
    exit 1
  fi

  # Check time since last heartbeat
  LAST_MOD=$(stat -c %Y "$HEARTBEAT_FILE")
  NOW=$(date +%s)
  DIFF=$((NOW - LAST_MOD))
  
  if [ $DIFF -gt 3600 ]; then
    echo "No heartbeat detected for 1 hour ($DIFF seconds). Shutting down."
    exit 0
  fi
done
