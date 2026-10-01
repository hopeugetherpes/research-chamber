#!/bin/bash
# Keeps live/watch.py running for good on the watch pod, restarting it if it ever exits. Start it on the pod with
#   (nohup bash live/watch_loop.sh Qwen_2.5_7B_instruct Llama_3.1_8B_instruct ... > /dev/null 2>&1 < /dev/null &)
# and follow /workspace/watch.log.
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
while true; do
  /workspace/venv/bin/python -u live/watch.py "$@" >> /workspace/watch.log 2>&1
  echo "$(date -u +%FT%TZ) watch exited with code $?, restarting in 30 s" >> /workspace/watch.log
  sleep 30
done
