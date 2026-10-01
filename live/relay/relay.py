"""Relay for the live games. live/watch.py on the GPU pod posts every token here, and every open page gets them
over one server-sent event stream, so all viewers see the same match at the same moment. It keeps nothing but the
match playing now in each game, for pages that join halfway.

  RELAY_SECRET=... uvicorn relay:app
"""
import asyncio
import hmac
import json
import os
import time

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

SECRET = os.environ["RELAY_SECRET"]
playing = {}   # game -> {"view", "turn", "text", "t"}
clients = set()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def broadcast(ev):
    for q in list(clients):
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:   # a page that stopped reading; it reconnects and gets a fresh snapshot
            clients.discard(q)


def snapshot():
    t = time.time()
    return {g: {**s, "age": round(t - s["t"], 1)} for g, s in playing.items()}


@app.post("/push")
async def push(request: Request, authorization: str = Header("")):
    if not hmac.compare_digest(authorization.encode(), f"Bearer {SECRET}".encode()):
        raise HTTPException(401)
    events = (await request.json())["events"]
    for ev in events:
        s = playing.setdefault(ev["game"], {"view": None, "turn": None, "text": "", "t": 0})
        if ev["type"] == "view":
            s.update(view=ev["view"], turn=None, text="")
        elif ev["type"] == "turn":
            s.update(turn=ev, text="")
        elif ev["type"] == "tok":
            s["text"] += ev["text"]
        s["t"] = time.time()
    broadcast({"type": "batch", "events": events})
    return {"viewers": len(clients)}


@app.get("/stream")
async def stream():
    q = asyncio.Queue(maxsize=2000)
    q.put_nowait({"type": "snapshot", "games": snapshot(), "viewers": len(clients) + 1})
    clients.add(q)
    broadcast({"type": "viewers", "n": len(clients)})

    async def events():
        try:
            while q in clients or not q.empty():
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            clients.discard(q)
            broadcast({"type": "viewers", "n": len(clients)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
def health():
    return {"viewers": len(clients), "age": {g: s["age"] for g, s in snapshot().items()}}
