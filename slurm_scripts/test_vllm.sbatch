#!/bin/bash
#SBATCH --job-name=test_vllm
#SBATCH --output=test-vllm-%j.out
#SBATCH --error=test-vllm-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a6000:1

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
export VLLM_HOST="127.0.0.1"
export VLLM_PORT=$(python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

echo "Starting vLLM server on port ${VLLM_PORT} for model ${MODEL_NAME}..."

cleanup_vllm() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    echo "Shutting down vLLM server (PID ${VLLM_PID})..."
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
  --max-model-len 32768 &
VLLM_PID=$!

python - <<'PY'
import json
import time
import urllib.request
import os
import sys

port = os.environ.get("VLLM_PORT")
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 600
last_error = None

print(f"Waiting for vLLM to become ready on port {port} (up to 10 mins)...")
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("data"):
                    print("vLLM server is READY.")
                    raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)

print(f"vLLM did not become ready on port {port}: {last_error}", file=sys.stderr)
sys.exit(1)
PY

# Check if vLLM process is still alive
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "vLLM process crashed before we could test it!" >&2
  exit 1
fi

echo "Sending test request to vLLM server..."
# Make a test chat completion request
curl -s -X POST "${OPENAI_BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d '{
    "model": "'"${MODEL_NAME}"'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello! Please reply with exactly: \"vLLM is working!\"" }
    ],
    "max_tokens": 50
  }' > test_vllm_response.json

echo -e "\nResponse received:"
cat test_vllm_response.json

echo -e "\n\nTest completed successfully."
exit 0
