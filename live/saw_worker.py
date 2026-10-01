#!/usr/bin/env python3
"""The Saw Test's three models on the GPU pod, beside the games. Each runs server.run_loop in its own thread and
posts its events to that model's server on Railway (live/server.py with PUSH_SECRET set), which streams them to
the page and saves every run.

  DEVICE=cuda PUSH_SECRET=... DATABASE_URL=... python live/saw_worker.py [model=url ...]

With arguments, only those models run, each posting to the url given.
"""
import os
import sys
import threading
import time

import httpx
import torch

import server

SERVERS = {"Qwen/Qwen3-4B": "https://chamber-production-dfc4.up.railway.app",
           "unsloth/Llama-3.2-3B-Instruct": "https://chamber-llama-production.up.railway.app",
           "microsoft/Phi-4-mini-instruct": "https://chamber-phi-production.up.railway.app"}


class Pusher(threading.Thread):
    """Posts one model's events to its server every 150 ms, and at least once a second to learn whether anyone is
    watching. Events that fail to send are kept and sent again, so every run still gets saved."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.head = f"{url}/push", {"Authorization": f"Bearer {os.environ['PUSH_SECRET']}"}
        self.lock, self.out, self.vector, self.viewers = threading.Lock(), [], None, 0

    def put(self, ev):
        with self.lock:
            self.out.append(ev)
            if ev["type"] == "vector":
                self.vector = ev

    def run(self):
        last, failing = 0.0, False
        with httpx.Client(timeout=10) as http:
            while True:
                time.sleep(0.15)
                with self.lock:
                    evs = list(self.out)
                if not evs and time.time() - last < 1:
                    continue
                try:
                    r = http.post(self.url, json={"events": evs}, headers=self.head)
                    r.raise_for_status()
                    got = r.json()
                except (httpx.HTTPError, ValueError) as e:
                    if not failing:
                        print(f"{self.url}: {e!r}, retrying", flush=True)
                    self.viewers, failing = 0, True
                    time.sleep(2)
                    continue
                if failing:
                    print(f"{self.url}: back", flush=True)
                with self.lock:
                    del self.out[:len(evs)]
                    if not got["vector"] and self.vector and self.vector not in self.out:   # the server restarted
                        self.out.append(self.vector)
                self.viewers, last, failing = got["viewers"], time.time(), False


def main():
    torch.set_num_threads(3)
    for model, url in (dict(a.split("=", 1) for a in sys.argv[1:]) or SERVERS).items():
        p = Pusher(url)
        p.start()
        threading.Thread(target=server.run_loop_or_exit, args=(model, p.put, lambda p=p: p.viewers > 0),
                         daemon=True).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
