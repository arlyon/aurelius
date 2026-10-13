#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-serve}" == migrate ]]; then exec /app/worker "$@"; fi
/app/llama-server --model /opt/model.gguf --alias Qwen3-Embedding-4B \
  --embedding --pooling last --host 127.0.0.1 --port 8081 \
  --ctx-size 4096 --batch-size 2048 --ubatch-size 2048 --parallel 1 \
  --threads 2 --threads-batch 2 --n-gpu-layers 0 --no-webui --log-disable &
model_pid=$!
worker_pid=''
cleanup() {
  [[ -z "$worker_pid" ]] || kill -TERM "$worker_pid" 2>/dev/null || true
  kill -TERM "$model_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT
ready=false
for ((i=0; i<150; i++)); do
  kill -0 "$model_pid" 2>/dev/null || exit 1
  if curl --silent --fail --max-time 2 http://127.0.0.1:8081/health >/dev/null; then ready=true; break; fi
  sleep 2
done
[[ "$ready" == true ]] || exit 1
printf '%s\n' 'Bundled CPU embedding model ready'
/app/worker "$@" &
worker_pid=$!
# Neither a dead adapter nor a dead worker may leave a healthy-looking container behind.
wait -n "$model_pid" "$worker_pid"
exit 1
