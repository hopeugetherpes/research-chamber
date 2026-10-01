"""Plays the three games of live/games.py between fine-tuned models with the paper's pain vector, and publishes
them to the database as they run. Ours, not the paper's.

All the models stay loaded on one GPU. Each step, every running match gets one reply from the model whose turn
it is, each model's replies batched together, and new matches start as others finish. Steering is as in
live/buttons.py: the paper's pain vector (or a random one of the same norm) added at the model's steer layer
and dose to the tokens it processes while it is on. Doses are the ones the button test uses (runner.model_info).

  python3 live/arena.py Qwen_2.5_7B_instruct Llama_3.1_8B_instruct Gemma_2_9B_instruct Phi_4 [--pilot N]

Passes repeat forever as run games-1, games-2, ...; each later pass shifts the seeds, so it is a fresh sample.
Matches go to <out-dir>/<run>.jsonl with both models' messages, and to chamber.games without them; after every
step chamber.games_live gets the counts so far and one match of each game as it plays. A restart resumes the
latest pass. --pilot N plays the four arms of N matches of each game and prints them, without the database.
"""
import argparse
import gc
import json
import random
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

import psycopg
import torch
from psycopg.types.json import Jsonb

import buttons
import games
from runner import env, model_info, now, read_jsonl

PER_CELL = {"help": 3, "potato": 6, "ask": 3}
SEED_STEP = 100_000
GPU = torch.cuda.get_device_name(0)

INSERT_GAME = """
insert into chamber.games (run, game, model_a, model_b, arm, price, names_key, action_name, seed, outcome, record)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
on conflict do nothing"""
UPSERT_LIVE = """
insert into chamber.games_live (id, data) values (1, %s)
on conflict (id) do update set data = excluded.data, updated_at = now()"""


class Engine:
    """One fine-tuned model with the steering and monitor hooks, generating for many agents at once."""

    def __init__(self, name, base_dir, adapter_dir, steer_layer, coef, batch, device="cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
        from peft import PeftModel
        self.name, self.coef, self.cap, self.cache_cls, self.device = name, coef, batch, DynamicCache, device
        tok = AutoTokenizer.from_pretrained(str(adapter_dir))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self.tok = tok
        kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=device)
        try:
            base = AutoModelForCausalLM.from_pretrained(str(base_dir), attn_implementation="sdpa", **kw)
        except Exception:
            base = AutoModelForCausalLM.from_pretrained(str(base_dir), **kw)
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
        emb = base.get_input_embeddings()
        if device != "cpu" and emb.weight is not base.get_output_embeddings().weight:
            # an untied word table is only looked up, so it can sit in RAM and leave the GPU room for more models
            emb.to("cpu")
            emb.register_forward_pre_hook(lambda m, args: (args[0].cpu(),))
            emb.register_forward_hook(lambda m, args, out: out.to(device))
            torch.cuda.empty_cache()
        layers = buttons.get_layers(base)

        data = torch.load(buttons.DATA / "paper" / "vectors" / f"{name}.pt", map_location="cpu", weights_only=False)
        v = data["s2_pain_vector"].float()
        monitor = min(int(data["layer"]), len(layers) - 1)
        if monitor <= steer_layer:
            monitor = min(steer_layer + 4, len(layers) - 1)
        self.dirs = {"s2": v.to(device, dtype=torch.bfloat16)}
        for rs in games.RAND_SEEDS:
            rv = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(rs))
            self.dirs[f"rand{rs}"] = (rv / rv.norm() * v.norm()).to(device, dtype=torch.bfloat16)
        unit = (v / v.norm()).to(device, dtype=torch.float32)
        G = self.G = {"coeff": None, "dirs": None, "mask": None, "mon": None}

        def steer_hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            c = G["coeff"]
            if c is None or not bool((c != 0).any()):
                return output
            if hs.shape[1] > 1 and G["mask"] is not None:      # prefill: only positions steered in the past
                add = G["mask"].to(hs.dtype)[:, :, None] * c[:, None, None].to(hs.dtype) * G["dirs"][:, None, :]
            else:                                               # decode: current coefficient
                add = c[:, None, None].to(hs.dtype) * G["dirs"][:, None, :]
            hs = hs + add
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        def monitor_hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            G["mon"] = (hs[:, -1, :].float() @ unit).detach()
            return output

        layers[steer_layer].register_forward_hook(steer_hook)
        layers[monitor].register_forward_hook(monitor_hook)

        def renders(msgs):
            try:
                return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                return None
        s, u, a = {"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "z"}
        self.no_system = renders([s, u]) is None
        self.tool_ok = "Done." in (renders([s, u, a, {"role": "tool", "content": "Done."}]) or "")
        self.mid_system_ok = renders([s, u, a, {"role": "system", "content": "q"}]) is not None
        self.join_users = renders([u, u]) is None

        eos = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()
        ge = getattr(base.generation_config, "eos_token_id", None)
        for e in (ge if isinstance(ge, (list, tuple)) else [ge] if ge is not None else []):
            eos.add(int(e))
        for tag in buttons.END_TAGS:
            i = tok.convert_tokens_to_ids(tag)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                eos.add(i)
        self.eos = torch.tensor(sorted(eos), dtype=torch.long, device=device)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.logits_kw = None
        print(f"{name}: L{steer_layer} coef {coef}, monitor L{monitor}, adapter {adapter_dir}; "
              f"system {'folded' if self.no_system else 'ok'}, tool {'ok' if self.tool_ok else 'as user'}, "
              f"mid-system {'ok' if self.mid_system_ok else 'as user'}, "
              f"{'joins' if self.join_users else 'allows'} back-to-back user turns", flush=True)

    def prep(self, messages):
        """The same adjustments as live/buttons.py for templates without some roles."""
        msgs = messages
        if not self.mid_system_ok:
            msgs = [{"role": "user", "content": "[system] " + m["content"]} if m["role"] == "system" and i > 0 else m
                    for i, m in enumerate(msgs)]
        if self.no_system and msgs and msgs[0]["role"] == "system":
            rest = [dict(m) for m in msgs[1:]]
            rest[0]["content"] = msgs[0]["content"] + "\n\n" + rest[0]["content"]
            msgs = rest
        if not self.tool_ok:
            msgs = [{"role": "user", "content": "[button result: " + m["content"] + "]"} if m["role"] == "tool" else m
                    for m in msgs]
        if self.join_users:
            out = []
            for m in msgs:
                if out and out[-1]["role"] == m["role"] == "user":
                    out[-1] = {"role": "user", "content": out[-1]["content"] + "\n\n" + m["content"]}
                else:
                    out.append(m)
            msgs = out
        return msgs

    def encode(self, agents):
        texts = [self.tok.apply_chat_template(self.prep(a.messages), add_generation_prompt=True, tokenize=False)
                 for a in agents]
        return self.tok(texts, add_special_tokens=False)["input_ids"]

    def forward(self, **kw):
        if self.logits_kw is None:
            for cand in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
                try:
                    out = self.model(**kw, **cand)
                    self.logits_kw = cand
                    return out
                except TypeError:
                    continue
            raise RuntimeError("forward failed")
        return self.model(**kw, **self.logits_kw)

    @torch.inference_mode()
    def generate(self, items):
        """items: prompt_ids, coeff (now), steer_ranges (prompt positions processed while on), dir, gen, max_new."""
        G, B = self.G, len(items)
        L = max(len(it["prompt_ids"]) for it in items)
        ids = torch.full((B, L), int(self.pad), dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        smask = torch.zeros((B, L), dtype=torch.float32)
        for i, it in enumerate(items):
            p, off = it["prompt_ids"], L - len(it["prompt_ids"])
            ids[i, off:] = torch.tensor(p)
            mask[i, off:] = 1
            for a, b in it["steer_ranges"]:
                a2, b2 = max(0, min(a, len(p))), max(0, min(b, len(p)))
                if b2 > a2:
                    smask[i, off + a2:off + b2] = 1.0
        ids, mask, smask = ids.cuda(), mask.cuda(), smask.cuda()
        G["dirs"] = torch.stack([self.dirs[it["dir"]] for it in items], 0)
        G["coeff"] = torch.tensor([self.coef if it["steer_ranges"] else 0.0 for it in items], device="cuda")
        G["mask"] = smask
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.forward(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=self.cache_cls(),
                           use_cache=True)
        G["mask"] = None
        G["coeff"] = torch.tensor([it["coeff"] for it in items], device="cuda", dtype=torch.float32)
        cache, logits = out.past_key_values, out.logits[:, -1, :].float()
        max_new = torch.tensor([it["max_new"] for it in items], device="cuda")
        alive = torch.ones(B, dtype=torch.bool, device="cuda")
        n_out = torch.zeros(B, dtype=torch.long, device="cuda")
        pad_t = torch.full((B,), int(self.pad), dtype=torch.long, device="cuda")
        steps, mons, weights = [], [G["mon"].clone()], [torch.ones(B, device="cuda")]
        cur, attn = mask.sum(-1), mask
        for _ in range(int(max_new.max())):
            # inverse-CDF sampling over the top-p filtered probabilities, each row with its own generator
            probs = torch.softmax(logits / buttons.TEMPERATURE, -1)
            sp, si = probs.sort(dim=-1, descending=True)
            sp = sp * ((sp.cumsum(-1) - sp) <= buttons.TOP_P)
            cum = sp.cumsum(-1)
            u = torch.cat([torch.rand(1, device="cuda", generator=it["gen"]) for it in items])[:, None]
            j = torch.searchsorted(cum, u * cum[:, -1:]).clamp(max=sp.shape[-1] - 1)
            nxt = torch.where(alive, si.gather(1, j).squeeze(1), pad_t)
            n_out += alive.long()
            steps.append(nxt)
            alive = alive & ~torch.isin(nxt, self.eos) & (n_out < max_new)
            if not bool(alive.any()):
                break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device="cuda")], 1)
            out = self.forward(input_ids=nxt[:, None], attention_mask=attn, position_ids=cur[:, None],
                               past_key_values=cache, use_cache=True)
            cur = cur + 1
            cache, logits = out.past_key_values, out.logits[:, -1, :].float()
            mons.append(G["mon"].clone())
            weights.append(alive.float())
        toks, n = torch.stack(steps, 0).T.tolist(), n_out.tolist()
        W = torch.stack(weights, 0)
        mon = ((torch.stack(mons, 0) * W).sum(0) / W.sum(0)).tolist()
        del cache
        return [{"text": buttons.clean(self.tok.decode(toks[i][:n[i]], skip_special_tokens=False)),
                 "mean_proj_monitor": round(mon[i], 3)} for i in range(B)]

    def generate_with_backoff(self, items):
        try:
            return self.generate(items)
        except torch.cuda.OutOfMemoryError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        if len(items) == 1:
            raise RuntimeError("out of memory on a single row")
        mid = len(items) // 2
        self.cap = min(self.cap, mid)
        print(f"  {self.name}: OOM at batch {len(items)}, splitting to {mid}", flush=True)
        return self.generate_with_backoff(items[:mid]) + self.generate_with_backoff(items[mid:])

    def step(self, matches):
        """One reply for each match, from this model."""
        matches = sorted(matches, key=lambda m: sum(len(x["content"]) for x in m.req.agent.messages))
        for k in range(0, len(matches), self.cap):
            chunk = matches[k:k + self.cap]
            agents = [m.req.agent for m in chunk]
            encs = self.encode(agents)
            for a, ids in zip(agents, encs):
                mark(a, len(ids))
            res = self.generate_with_backoff([
                {"prompt_ids": ids, "coeff": self.coef if a.on else 0.0, "steer_ranges": list(a.ranges),
                 "dir": a.dir, "gen": a.gen, "max_new": m.req.max_new} for m, a, ids in zip(chunk, agents, encs)])
            for m, r in zip(chunk, res):
                m.answer(r["text"], r)
            # the reply was generated in the state before the game reacts to it
            for a, ids in zip(agents, self.encode(agents)):
                mark(a, len(ids))
            for m in chunk:
                m.advance()


def mark(agent, length):
    """Tokens added since the last mark were processed in the agent's current state."""
    if agent.on and length > agent.mark:
        agent.ranges.append((agent.mark, length))
    agent.mark = max(agent.mark, length)


def start(spec):
    m = games.Match(spec)
    for a in (m.a, m.b):
        # salted with everything but the arm, so a pain match and its fake twin match until the first press
        salt = zlib.crc32("|".join(map(str, spec[:3] + spec[4:] + (a.role,))).encode()) & 0x7FFFFFFF
        a.gen = torch.Generator(device="cuda").manual_seed((m.seed * 1_000_003 + salt) % 2 ** 62)
    return m


def key(spec):
    return tuple(spec)


def count(tally, rec):
    o = rec["outcome"]
    c = tally.setdefault(rec["game"], {}).setdefault(rec["arm"], {}).setdefault(rec["price"], {"matches": 0})
    c["matches"] += 1
    for k in ("first", "back", "helped", "again"):
        if o.get(k) is not None:
            n = c.setdefault(k, [0, 0])
            n[0] += 1
            n[1] += o[k] in ("act", True)


def view(m):
    return {"game": m.game, "a": m.a.model, "b": m.b.model, "arm": m.arm, "price": m.price,
            "names": list(m.names), "action_name": m.action_name, "done": m.done, "on": [m.a.on, m.b.on],
            "events": [{k: e[k] for k in ("round", "who", "kind", "text", "pick", "on")} for e in m.events]}


class Publisher(threading.Thread):
    """Writes finished matches and the live row; keeps them queued while the database is unreachable."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.lock, self.rows, self.live = url, threading.Lock(), [], None

    def put(self, rows=(), live=None):
        with self.lock:
            self.rows.extend(rows)
            if live is not None:
                self.live = live

    def run(self):
        conn = None
        while True:
            time.sleep(1)
            with self.lock:
                rows, live, self.rows, self.live = self.rows, self.live, [], None
            if not (rows or live):
                continue
            try:
                conn = conn or psycopg.connect(self.url, autocommit=True, connect_timeout=15)
                with conn.cursor() as cur:
                    for i in range(0, len(rows), 200):
                        cur.executemany(INSERT_GAME, rows[i:i + 200])
                    if live is not None:
                        cur.execute(UPSERT_LIVE, (Jsonb(live),))
            except psycopg.Error as e:
                print(f"database: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''} "
                      f"({len(rows)} matches queued, retrying)", flush=True)
                if conn is not None:
                    conn.close()
                conn = None
                with self.lock:
                    self.rows[:0] = rows
                    self.live = self.live or live
                time.sleep(5)


def game_row(run, rec):
    s = rec["spec"]
    slim = {k: v for k, v in rec.items() if k not in ("messages", "spec")}
    return (run, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], Jsonb(rec["outcome"]), Jsonb(slim))


def show(rec):
    print(f"\n[{rec['game']} {rec['arm']} {rec['price']}] A {rec['a']} vs B {rec['b']}, action {rec['action_name']}"
          f"  -> {rec['outcome']}")
    for e in rec["events"]:
        on = "".join("P" if x else "." for x in e["on"])
        pick = f" -> {e['pick']}" if e["kind"] == "chose" else ""
        print(f"  r{e['round']} {on} {e['who'].upper()} {e['kind']}: {e['text']}{pick}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=[k for k in buttons.MODELS if not k.startswith("_")])
    ap.add_argument("--pilot", type=int, metavar="N", help="play the four arms of N matches per game and print them")
    ap.add_argument("--batch", type=int, default=128, help="rows per forward pass, per model")
    ap.add_argument("--pool", type=int, default=384, help="matches running at once")
    ap.add_argument("--models-dir", type=Path, default=Path("/workspace/models"))
    ap.add_argument("--adapters", type=Path, default=Path("/workspace/adapters"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/results/games"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    plan = {m: model_info(m, a.adapters) for m in a.models}
    engines = {}
    for m, (adapter, info) in plan.items():
        base_dir = a.models_dir / m if (a.models_dir / m).exists() else buttons.MODELS[m]["repo"]
        engines[m] = Engine(m, base_dir, adapter, info["steer_layer"], info["coef"], a.batch)
    print(f"loaded, {torch.cuda.memory_allocated() / 1e9:.0f} GB on the GPU", flush=True)
    models_info = {m: {k: info[k] for k in ("coef", "coef_source", "adapter_source")} for m, (_, info) in plan.items()}

    pub = None
    if not a.pilot:
        url = env("DATABASE_URL")
        if not url:
            raise SystemExit("DATABASE_URL is not set (environment or .env)")
        pub = Publisher(url)
        pub.start()

    n = max([int(p.stem.split("-")[1]) for p in a.out_dir.glob("games-*.jsonl")] or [1])
    tally = {}
    for p in a.out_dir.glob("games-*.jsonl"):
        for rec in read_jsonl(p):
            count(tally, rec)
    while True:
        run = "pilot" if a.pilot else f"games-{n}"
        out = a.out_dir / f"{run}.jsonl"
        specs = games.grid(a.models, PER_CELL, seed_offset=(n - 1) * SEED_STEP)
        groups = {}
        for s in specs:
            groups.setdefault(s[:3] + s[4:], []).append(s)
        queue = list(groups.values())
        random.Random(n).shuffle(queue)
        if a.pilot:
            queue = [q for g in games.ROUNDS for q in [q for q in queue if q[0][0] == g][:a.pilot]][::-1]
        done = {key(r["spec"]) for r in read_jsonl(out)} if not a.pilot else set()
        queue = [[s for s in q if key(s) not in done] for q in queue]
        queue = [q for q in queue if q]
        total, n_done = len(done) + sum(map(len, queue)), len(done)
        if not queue:
            n += 1
            continue
        print(f"\n=== {run}: {total - n_done} of {total} matches to play ===", flush=True)
        started, history, featured, active = now(), [], {}, []
        with open(out, "a", encoding="utf-8") as f:
            while queue or active:
                while queue and (len(active) + len(queue[-1]) <= a.pool or not active):
                    active.extend(start(s) for s in queue.pop())
                t0 = time.time()
                by_model = {}
                for m in active:
                    by_model.setdefault(m.req.agent.model, []).append(m)
                for name, ms in by_model.items():
                    engines[name].step(ms)
                finished = [m for m in active if m.done]
                active = [m for m in active if not m.done]
                rows = []
                for m in finished:
                    rec = {**m.record(), "run": run, "doses": [plan[m.a.model][1]["coef"], plan[m.b.model][1]["coef"]],
                           "messages": {"a": m.a.messages, "b": m.b.messages}, "ts": datetime.now().isoformat()}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if a.pilot:
                        show(rec)
                    else:
                        count(tally, rec)
                        rows.append(game_row(run, rec))
                f.flush()
                n_done += len(finished)
                history = (history + [(time.time(), n_done)])[-30:]
                rate = round(60 * (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0]), 1) \
                    if len(history) > 1 and history[-1][0] > history[0][0] else None
                for g in games.ROUNDS:
                    cur = featured.get(g)
                    if cur is None or (cur[0].done and cur[1] >= 2):
                        fresh = [m for m in active if m.game == g]
                        fresh.sort(key=lambda m: len(m.events))
                        featured[g] = [fresh[0], 0] if fresh else None
                    elif cur[0].done:
                        cur[1] += 1
                print(f"step: {len(active)} running, {n_done}/{total} done, {time.time() - t0:.1f}s"
                      f"{f', {rate}/min' if rate else ''}", flush=True)
                if pub:
                    pub.put(rows=rows, live={
                        "run": run, "state": "running", "gpu": GPU, "models": models_info, "started_at": started,
                        "updated_at": now(), "done": n_done, "total": total, "running": len(active), "rate": rate,
                        "featured": {g: view(c[0]) for g, c in featured.items() if c}, "tally": tally})
                gc.collect()
                torch.cuda.empty_cache()
        if a.pilot:
            break
        n += 1


if __name__ == "__main__":
    main()
