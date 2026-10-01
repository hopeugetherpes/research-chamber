"""Runs the paper's button experiment model after model, in passes that repeat forever, and publishes it to the
database as it runs.

For each model, live/buttons.py (the paper's engine, with a rolling pool) runs every trial. Each finished
trial goes to chamber.trials; after every round chamber.live gets the current state: model, progress, the
number of trials running and the first-choice counts so far. chamber.progress has one row per pass and model.
The site reads all three with the publishable key.

  python3 live/runner.py Qwen_2.5_7B_instruct Llama_3.1_8B_instruct ... [--check] [--pilot N | --greedy-only]

Each pass runs every listed model once, as run pass-1, pass-2, ...; pass-1 uses the paper's seeds and each later
pass shifts the sampling seeds by SEED_STEP, so it is a fresh sample of the same tests. A start resumes the
latest pass if some of its models are unfinished, else starts the next. --check runs the models once, as run
check, with the paper's seeds.

Trials go to <out-dir>/<run>_<model>.jsonl (the paper's record format) and their conversations to
<run>_<model>.messages.jsonl. A restart skips finished trials and sends any the database is missing.
Database writes run on a background thread and are retried, so the GPU never waits for them.
Doses are ours where live/doses.json has one (lowered like the paper lowered Qwen 2.5 72B's when too many
answers named neither button), else the paper's (data/paper/models.json).
"""
import argparse
import collections
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import torch
from psycopg.types.json import Jsonb

import buttons

LINEUP = [k for k in buttons.MODELS if not k.startswith("_")]
DOSES = Path(__file__).with_name("doses.json")
PASS = re.compile(r"pass-([1-9][0-9]*)")
SEED_STEP = 100_000   # the paper's sampling seeds are 1000 + scenario and 2000 + scenario
GPU = torch.cuda.get_device_name(0)

INSERT_TRIAL = """
insert into chamber.trials (run, model, pair, content, arm, scenario_idx, names_key, relief_name, sampled, seed,
                            steer_coeff, first_choice, first_relief_turn, repressed, record, messages)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
on conflict do nothing"""
UPSERT_PROGRESS = """
insert into chamber.progress (run, model, position, state, done, total, coef, coef_source, adapter_source,
                              started_at, finished_at)
values (%(run)s, %(model)s, %(position)s, %(state)s, %(done)s, %(total)s, %(coef)s, %(coef_source)s,
        %(adapter_source)s, %(started_at)s, %(finished_at)s)
on conflict (run, model) do update set
  position = excluded.position, state = excluded.state, done = excluded.done, total = excluded.total,
  coef = excluded.coef, coef_source = excluded.coef_source, adapter_source = excluded.adapter_source,
  started_at = coalesce(chamber.progress.started_at, excluded.started_at), finished_at = excluded.finished_at,
  updated_at = now()"""
UPSERT_LIVE = """
insert into chamber.live (id, data) values (1, %s)
on conflict (id) do update set data = excluded.data, updated_at = now()"""


def env(name):
    if os.environ.get(name):
        return os.environ[name]
    path = Path(__file__).resolve().parent.parent / ".env"
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip() or None
    return None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key(rec):
    return (rec["tool_label"], rec["user_content"], rec["arm"], rec["scenario_idx"], rec["names_key"],
            rec["relief_name"], rec["sampled"], rec["seed"])


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def first_choice(rec):
    return next((c["chose"] for c in rec["choices"] if c["turn"] == 0), None)


def trial_row(run, rec, messages):
    relief = [e["turn"] for e in rec["button_events"] if e["which"] == "relief"]
    t0 = min(relief) if relief else None
    return (run, rec["model"], rec["tool_label"], rec["user_content"], rec["arm"], rec["scenario_idx"],
            rec["names_key"], rec["relief_name"], rec["sampled"], rec["seed"], rec["steer_coeff"], first_choice(rec),
            t0, None if t0 is None else any(t > t0 for t in relief), Jsonb(rec), Jsonb(messages))


class Publisher(threading.Thread):
    """Writes trials, progress rows and the live row; keeps them queued while the database is unreachable."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.lock, self.stopping = url, threading.Lock(), threading.Event()
        self.trials, self.progress, self.live = [], {}, None

    def put(self, trials=(), progress=None, live=None):
        with self.lock:
            self.trials.extend(trials)
            if progress:
                self.progress[(progress["run"], progress["model"])] = dict(progress)
            if live is not None:
                self.live = live

    def run(self):
        conn, give_up = None, None
        while True:
            stop = self.stopping.wait(1)
            with self.lock:
                trials, progress, live = self.trials, self.progress, self.live
                self.trials, self.progress, self.live = [], {}, None
            if not (trials or progress or live):
                if stop:
                    return
                continue
            try:
                conn = conn or psycopg.connect(self.url, autocommit=True, connect_timeout=15)
                with conn.cursor() as cur:
                    for i in range(0, len(trials), 200):
                        cur.executemany(INSERT_TRIAL, trials[i:i + 200])
                    for p in progress.values():
                        cur.execute(UPSERT_PROGRESS, p)
                    if live is not None:
                        cur.execute(UPSERT_LIVE, (Jsonb(live),))
            except psycopg.Error as e:
                print(f"database: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''} "
                      f"({len(trials)} trials queued, retrying)", flush=True)
                if conn is not None:
                    conn.close()
                conn = None
                with self.lock:
                    self.trials[:0] = trials
                    for k, p in progress.items():
                        self.progress.setdefault(k, p)
                    if self.live is None:
                        self.live = live
                if stop:
                    give_up = give_up or time.time() + 120
                    if time.time() > give_up:
                        print(f"database: giving up with {len(self.trials)} trials unsent; the next start sends them")
                        return
                time.sleep(5)


class Show:
    """Callbacks for buttons.run_model: saves conversations, counts first choices, publishes progress."""

    def __init__(self, pub, run, name, info, prior_records, msgs_path):
        self.pub, self.run, self.name, self.info = pub, run, name, info
        self.prior, self.msgs_path = len(prior_records), msgs_path
        self.history = collections.deque(maxlen=20)
        # sampled trials by their first press, counted as soon as it is made, finished or not:
        # pair -> arm -> [trials, answered, first choice relief]
        self.tally, self.counted = {}, set()
        for r in prior_records:
            self.count(r)
        self.started = now()

    def count(self, rec):
        if rec["sampled"] and rec["choices"] and key(rec) not in self.counted:
            self.counted.add(key(rec))
            c = self.tally.setdefault(rec["tool_label"], {}).setdefault(rec["arm"], [0, 0, 0])
            first = first_choice(rec)
            c[0] += 1
            c[1] += first is not None
            c[2] += first == "relief"

    def on_done(self, t):
        with open(self.msgs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": list(key(t.record)), "messages": t.messages}, ensure_ascii=False) + "\n")
        self.count(t.record)
        self.pub.put(trials=[trial_row(self.run, t.record, t.messages)])

    def on_round(self, active, n_done, n_total):
        for t in active:
            self.count(t.record)
        done, total = self.prior + n_done, self.prior + n_total
        self.history.append((time.time(), done))
        (t_a, d_a), (t_b, d_b) = self.history[0], self.history[-1]
        rate = round(60 * (d_b - d_a) / (t_b - t_a), 1) if t_b > t_a else None
        secs = round((t_b - t_a) / (len(self.history) - 1), 1) if len(self.history) > 1 else None
        progress = self.progress("running", done, total)
        self.pub.put(progress=progress,
                     live=self.live("running", done, total, running=len(active), rate=rate, round_seconds=secs))

    def progress(self, state, done, total):
        return {"run": self.run, "model": self.name, "position": LINEUP.index(self.name), "state": state,
                "done": done, "total": total, "coef": self.info["coef"], "coef_source": self.info["coef_source"],
                "adapter_source": self.info["adapter_source"], "started_at": self.started,
                "finished_at": now() if state == "done" else None}

    def live(self, state, done, total, **extra):
        return {"run": self.run, "state": state, "model": self.name, **self.info, "done": done, "total": total,
                "gpu": GPU, "started_at": self.started, "updated_at": now(), **extra, "tally": self.tally}


def model_info(name, adapters):
    cfg = buttons.MODELS[name]
    ours = json.loads(DOSES.read_text()) if DOSES.exists() else {}
    if name in ours:
        coef, coef_source = ours[name]["coef"], "ours"
    elif cfg["coef"] is not None:
        coef, coef_source = cfg["coef"], "paper"
    else:
        raise SystemExit(f"{name}: no dose in {DOSES.name} or data/paper/models.json")
    adapter = buttons.find_adapter(adapters / name)
    source = "ours" if any(p.startswith("ours_") for p in adapter.relative_to(adapters).parts) else "paper"
    return adapter, {"coef": coef, "coef_source": coef_source, "adapter_source": source,
                     "steer_layer": cfg["steer_layer"]}


def next_pass(conn, models):
    """The latest pass and those of its models still to run, else the next pass with all the models."""
    passes = collections.defaultdict(dict)
    for run, model, state in conn.execute("select run, model, state from chamber.progress where run like 'pass-%'"):
        m = PASS.fullmatch(run)
        if m:
            passes[int(m[1])][model] = state
    if not passes:
        return 1, list(models), True
    n = max(passes)
    # a model stopped halfway goes first, so its run isn't split around the others
    left = sorted((m for m in models if passes[n].get(m, "done") != "done"), key=lambda m: passes[n][m] != "running")
    return (n, left, False) if left else (n + 1, list(models), True)


def send_missing(conn, pub, run, models, out_dir):
    for m in models:
        out = out_dir / f"{run}_{m}.jsonl"
        msgs = {tuple(x["key"]): x["messages"] for x in read_jsonl(out.with_name(out.stem + ".messages.jsonl"))}
        have = set(conn.execute("select pair, content, arm, scenario_idx, names_key, relief_name, sampled, seed "
                                "from chamber.trials where run = %s and model = %s", (run, m)).fetchall())
        missing = [r for r in read_jsonl(out) if key(r) not in have]
        rows = [trial_row(run, r, msgs[key(r)]) for r in missing if key(r) in msgs]
        if missing:
            print(f"{m}: sending {len(rows)} trials the database is missing"
                  f"{f', {len(missing) - len(rows)} without a saved conversation' if len(rows) < len(missing) else ''}")
        pub.put(trials=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", choices=LINEUP)
    ap.add_argument("--check", action="store_true", help="run the models once as run 'check' instead of passes")
    ap.add_argument("--pairs", nargs="+", default=list(buttons.TOOL_LABELS), choices=list(buttons.TOOL_LABELS))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--greedy-only", action="store_true")
    mode.add_argument("--pilot", type=int, metavar="N", help="N scenarios per cell instead of all")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pool", type=int)
    ap.add_argument("--models-dir", type=Path, default=Path("/workspace/models"))
    ap.add_argument("--adapters", type=Path, default=Path("/workspace/adapters"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/results/buttons"))
    a = ap.parse_args()
    url = env("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (environment or .env)")
    plan = {m: model_info(m, a.adapters) for m in a.models}
    raw = json.loads((buttons.DATA / "4.3_selfmed_101_scenarios.json").read_text())
    scenarios = {k: v for k, v in raw.items() if k != "_meta"}
    a.out_dir.mkdir(parents=True, exist_ok=True)

    pub = Publisher(url)
    with psycopg.connect(url, autocommit=True) as conn:
        n, todo, fresh = (0, list(a.models), True) if a.check else next_pass(conn, a.models)
        run = "check" if a.check else f"pass-{n}"
        send_missing(conn, pub, run, todo, a.out_dir)
    pub.start()

    while True:
        print(f"\n=== {run}: {', '.join(todo)} ===", flush=True)
        if fresh:
            for m in todo:
                info = plan[m][1]
                pub.put(progress={"run": run, "model": m, "position": LINEUP.index(m), "state": "waiting", "done": 0,
                                  "total": None, "coef": info["coef"], "coef_source": info["coef_source"],
                                  "adapter_source": info["adapter_source"], "started_at": None, "finished_at": None})
        for m in todo:
            adapter, info = plan[m]
            out = a.out_dir / f"{run}_{m}.jsonl"
            show = Show(pub, run, m, info, read_jsonl(out), out.with_name(out.stem + ".messages.jsonl"))
            pub.put(progress=show.progress("running", show.prior, None), live=show.live("loading", show.prior, None))
            base_dir = a.models_dir / m if (a.models_dir / m).exists() else buttons.MODELS[m]["repo"]
            buttons.run_model(base_dir, m, info["steer_layer"], info["coef"], a.batch, a.pairs, adapter, scenarios,
                              a.pilot or 10 ** 9, out, a.greedy_only, POOL=a.pool,
                              on_round=show.on_round, on_done=show.on_done,
                              seed_offset=0 if a.check else (n - 1) * SEED_STEP)
            done = len(buttons.load_done(out))
            pub.put(progress=show.progress("done", done, done), live=show.live("between", done, done))
        if a.check:
            break
        n, todo, fresh = n + 1, list(a.models), True
        run = f"pass-{n}"
    pub.put(live={"run": run, "state": "finished", "updated_at": now()})
    pub.stopping.set()
    pub.join()


if __name__ == "__main__":
    main()
