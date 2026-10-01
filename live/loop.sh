#!/bin/bash
# Keeps live/runner.py running for good: the runner loops over the passes itself, and this restarts it if it
# ever exits (it resumes the unfinished pass). Start it on the pod with
#   (nohup bash live/loop.sh Qwen_2.5_7B_instruct Llama_3.1_8B_instruct ... > /dev/null 2>&1 < /dev/null &)
# and follow /workspace/loop.log.
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=/workspace/hf
while true; do
  /workspace/venv/bin/python -u live/runner.py "$@" --batch 256 --pool 512 >> /workspace/loop.log 2>&1
  echo "$(date -u +%FT%TZ) runner exited with code $?, restarting in 60 s" >> /workspace/loop.log
  sleep 60
done
