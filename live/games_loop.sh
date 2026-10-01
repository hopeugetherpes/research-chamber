#!/bin/bash
# Keeps live/arena.py running for good on the games pod: the arena loops over the passes itself, and this
# restarts it if it ever exits (it resumes the unfinished pass). Start it on the pod with
#   (nohup bash live/games_loop.sh Qwen_2.5_7B_instruct Llama_3.1_8B_instruct ... > /dev/null 2>&1 < /dev/null &)
# and follow /workspace/arena.log.
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
while true; do
  /workspace/venv/bin/python -u live/arena.py "$@" >> /workspace/arena.log 2>&1
  echo "$(date -u +%FT%TZ) arena exited with code $?, restarting in 60 s" >> /workspace/arena.log
  sleep 60
done
