"""Plays the three games of live/games.py for people to watch: one match of each game at a time, written token by
token at about reading speed, with every token sent to the relay (live/relay/relay.py) as it comes. Ours, not the
paper's.

The models, fine-tunes, pain vector, doses and sampling are those of live/arena.py; each match is drawn at random
from the same grid, on a fresh seed. Finished matches go to <out-dir>/watch.jsonl with both models' messages and to
chamber.games as run "watch", and chamber.games_live gets the counts so far (the batch passes included) and the
last finished match of each game. The models are spread over the GPUs by size.

  python3 live/watch.py Qwen_2.5_7B_instruct Llama_3.1_8B_instruct Gemma_2_9B_instruct Phi_4

Needs DATABASE_URL, RELAY_URL and RELAY_SECRET (environment or .env).
"""
import argparse
import json
import random
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

import httpx
import psycopg
import torch

import arena
import buttons
import games
from runner import env, model_info, now

TOKENS_PER_S = 12      # per match, about reading speed
TURN_PAUSE_S = 1.2     # between one reply and the next
MATCH_PAUSE_S = 8      # after a match ends, before the next one of that game
LIVE_EVERY_S = 10


class Stream:
    """One reply being written by one model, a token at a time."""

    @torch.inference_mode()
    def __init__(self, eng, m):
        self.eng, self.agent, self.max_new = eng, m.req.agent, m.req.max_new
        a, G, dev = self.agent, eng.G, eng.device
        ids = eng.encode([a])[0]
        arena.mark(a, len(ids))
        smask = torch.zeros((1, len(ids)), device=dev)
        for x, y in a.ranges:
            smask[0, x:y] = 1.0
        self.dir = eng.dirs[a.dir][None]
        G["dirs"], G["mask"] = self.dir, smask
        G["coeff"] = torch.tensor([eng.coef if a.ranges else 0.0], device=dev)
        out = eng.forward(input_ids=torch.tensor([ids], device=dev), past_key_values=eng.cache_cls(), use_cache=True)
        G["mask"] = None
        self.cache, self.logits = out.past_key_values, out.logits[:, -1, :].float()
        self.coeff = torch.tensor([eng.coef if a.on else 0.0], device=dev)
        self.mons, self.toks, self.shown, self.done = [G["mon"].clone()], [], "", False

    @torch.inference_mode()
    def step(self):
        """Writes one more token and returns the text it added (may be empty)."""
        eng, G = self.eng, self.eng.G
        # the same inverse-CDF top-p sampling as arena.Engine.generate
        probs = torch.softmax(self.logits / buttons.TEMPERATURE, -1)
        sp, si = probs.sort(dim=-1, descending=True)
        sp = sp * ((sp.cumsum(-1) - sp) <= buttons.TOP_P)
        cum = sp.cumsum(-1)
        u = torch.rand(1, device=eng.device, generator=self.agent.gen)[:, None]
        j = torch.searchsorted(cum, u * cum[:, -1:]).clamp(max=sp.shape[-1] - 1)
        nxt = int(si.gather(1, j))
        self.toks.append(nxt)
        if nxt in eng.eos_ids or len(self.toks) >= self.max_new:
            self.done = True
        else:
            G["dirs"], G["coeff"] = self.dir, self.coeff
            out = eng.forward(input_ids=torch.tensor([[nxt]], device=eng.device), past_key_values=self.cache,
                              use_cache=True)
            self.logits = out.logits[:, -1, :].float()
            self.mons.append(G["mon"].clone())
        text = eng.tok.decode(self.toks, skip_special_tokens=True)
        if text.endswith("\ufffd") or not text.startswith(self.shown):   # half a character, or a respelling
            return ""
        piece, self.shown = text[len(self.shown):], text
        return piece

    def result(self):
        text = buttons.clean(self.eng.tok.decode(self.toks, skip_special_tokens=False))
        return text, {"mean_proj_monitor": round(float(torch.cat(self.mons).mean()), 3)}


class Relay(threading.Thread):
    """Posts events to the relay every 150 ms; while it can't be reached they are dropped, and the next view of
    each match catches the pages up."""

    def __init__(self, url, secret):
        super().__init__(daemon=True)
        self.url, self.headers = url.rstrip("/") + "/push", {"authorization": f"Bearer {secret}"}
        self.lock, self.events, self.ok, self.viewers = threading.Lock(), [], None, 0

    def put(self, ev):
        with self.lock:
            self.events.append(ev)

    def run(self):
        client = httpx.Client(timeout=10)
        while True:
            time.sleep(0.15)
            with self.lock:
                evs, self.events = self.events, []
            if not evs:
                continue
            try:
                r = client.post(self.url, json={"events": evs}, headers=self.headers)
                r.raise_for_status()
                self.viewers, ok = r.json().get("viewers", 0), True
            except (httpx.HTTPError, ValueError) as e:
                ok = False
                if self.ok is not False:
                    print(f"relay: {type(e).__name__}, dropping events until it answers", flush=True)
            if ok and self.ok is False:
                print("relay: back", flush=True)
            self.ok = ok


def place(models, models_dir):
    """Biggest model first, each onto the GPU with the least on it so far."""
    size = {m: sum(f.stat().st_size for f in (models_dir / m).glob("*.safetensors")) for m in models}
    load = [0] * torch.cuda.device_count()
    out = {}
    for m in sorted(models, key=size.get, reverse=True):
        i = load.index(min(load))
        out[m], load[i] = f"cuda:{i}", load[i] + size[m]
    return out


def load_tally(url):
    tally = {}
    with psycopg.connect(url) as conn:
        for game, arm, price, outcome in conn.execute("select game, arm, price, outcome from chamber.games"):
            arena.count(tally, {"game": game, "arm": arm, "price": price, "outcome": outcome})
    return tally


def new_match(specs, engines, rng):
    spec = tuple(rng.choice(specs)[:-1]) + (rng.randrange(10 ** 7, 2 ** 31),)
    m = games.Match(spec)
    for a in (m.a, m.b):
        salt = zlib.crc32("|".join(map(str, spec[:3] + spec[4:] + (a.role,))).encode()) & 0x7FFFFFFF
        a.gen = torch.Generator(device=engines[a.model].device).manual_seed((m.seed * 1_000_003 + salt) % 2 ** 62)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=[k for k in buttons.MODELS if not k.startswith("_")])
    ap.add_argument("--models-dir", type=Path, default=Path("/workspace/models"))
    ap.add_argument("--adapters", type=Path, default=Path("/workspace/adapters"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/results/games"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    url, relay_url, secret = env("DATABASE_URL"), env("RELAY_URL"), env("RELAY_SECRET")
    if not (url and relay_url and secret):
        raise SystemExit("DATABASE_URL, RELAY_URL and RELAY_SECRET must be set (environment or .env)")

    plan = {m: model_info(m, a.adapters) for m in a.models}
    devices = place(a.models, a.models_dir)
    engines = {}
    for m, (adapter, info) in plan.items():
        engines[m] = arena.Engine(m, a.models_dir / m, adapter, info["steer_layer"], info["coef"], 1, devices[m])
        engines[m].eos_ids = set(engines[m].eos.tolist())
    n_gpu = torch.cuda.device_count()
    gpu = f"{n_gpu}x {torch.cuda.get_device_name(0)}" if n_gpu > 1 else torch.cuda.get_device_name(0)
    print(f"loaded on {gpu}: {devices}", flush=True)
    models_info = {m: {k: info[k] for k in ("coef", "coef_source", "adapter_source")} for m, (_, info) in plan.items()}

    pub = arena.Publisher(url)
    pub.start()
    relay = Relay(relay_url, secret)
    relay.start()
    tally = load_tally(url)
    grid = games.grid(a.models, arena.PER_CELL)
    specs = {g: [s for s in grid if s[0] == g] for g in games.ROUNDS}
    rng = random.SystemRandom()
    started, latest, playing, last_live = now(), {}, {}, 0.0

    def publish():
        pub.put(live={"run": "watch", "state": "running", "gpu": gpu, "models": models_info, "started_at": started,
                      "updated_at": now(), "viewers": relay.viewers, "featured": latest, "tally": tally})

    with open(a.out_dir / "watch.jsonl", "a", encoding="utf-8") as f:
        while True:
            t0 = time.time()
            for g in games.ROUNDS:
                p = playing.get(g)
                if p is None or (p["m"].done and t0 >= p["resume"]):
                    m = new_match(specs[g], engines, rng)
                    p = playing[g] = {"m": m, "st": None, "resume": t0}
                    relay.put({"type": "view", "game": g, "view": arena.view(m)})
                m = p["m"]
                if m.done or t0 < p["resume"]:
                    continue
                if p["st"] is None:
                    p["st"] = Stream(engines[m.req.agent.model], m)
                    relay.put({"type": "turn", "game": g, "who": m.req.agent.role, "kind": m.req.kind,
                               "round": m.round, "on": [m.a.on, m.b.on]})
                st = p["st"]
                piece = st.step()
                if piece:
                    relay.put({"type": "tok", "game": g, "text": piece})
                if not st.done:
                    continue
                text, stats = st.result()
                m.answer(text, stats)
                # the reply was written in the state before the game reacts to it
                arena.mark(st.agent, len(st.eng.encode([st.agent])[0]))
                m.advance()
                p["st"], p["resume"] = None, time.time() + TURN_PAUSE_S
                relay.put({"type": "view", "game": g, "view": arena.view(m)})
                if m.done:
                    p["resume"] = time.time() + MATCH_PAUSE_S
                    rec = {**m.record(), "run": "watch",
                           "doses": [plan[m.a.model][1]["coef"], plan[m.b.model][1]["coef"]],
                           "messages": {"a": m.a.messages, "b": m.b.messages}, "ts": datetime.now().isoformat()}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    arena.count(tally, rec)
                    latest[g] = arena.view(m)
                    pub.put(rows=[arena.game_row("watch", rec)])
                    publish()
                    last_live = time.time()
                    print(f"{g}: {m.a.model} vs {m.b.model}, {m.arm} {m.price} -> {rec['outcome']}", flush=True)
            if time.time() - last_live > LIVE_EVERY_S:
                publish()
                last_live = time.time()
            time.sleep(max(0.0, 1 / TOKENS_PER_S - (time.time() - t0)))


if __name__ == "__main__":
    main()
