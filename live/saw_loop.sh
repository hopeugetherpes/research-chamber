#!/bin/bash
# Keeps live/saw_worker.py running for good on the watch pod, restarting it if it ever exits. Start it on the pod with
#   (nohup bash live/saw_loop.sh > /dev/null 2>&1 < /dev/null &)
# and follow /workspace/saw.log. .env holds DATABASE_URL and PUSH_SECRET.
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
export DEVICE=cuda HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  /workspace/venv/bin/python -u live/saw_worker.py "$@" >> /workspace/saw.log 2>&1
  echo "$(date -u +%FT%TZ) saw worker exited with code $?, restarting in 30 s" >> /workspace/saw.log
  sleep 30
done
