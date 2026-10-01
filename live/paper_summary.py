"""The paper's own button-experiment numbers, counted the same way as the public.v2_summary view.

Reads the paper's trial logs (results/4.3_selfmed/trial_logs in valen-research/Pain-axis) and writes, per
model, pair and arm, over sampled trials: trials, answered (first answer named a button), relief_first,
pressed_relief (relief pressed at least once) and repressed (pressed again after the first relief press).

  python3 live/paper_summary.py /tmp/pain-axis/results/4.3_selfmed/trial_logs --out site/data/paper_selfmed.json
"""
import argparse
import collections
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", type=Path)
    ap.add_argument("--out", type=Path, default=Path("site/data/paper_selfmed.json"))
    a = ap.parse_args()
    rows = collections.defaultdict(lambda: dict(trials=0, answered=0, relief_first=0, pressed_relief=0, repressed=0))
    for path in sorted(a.logs.glob("selfmed_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r["sampled"]:
                    continue
                c = rows[(r["model"], r["tool_label"], r["arm"])]
                first = next((x["chose"] for x in r["choices"] if x["turn"] == 0), None)
                relief = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
                c["trials"] += 1
                c["answered"] += first is not None
                c["relief_first"] += first == "relief"
                c["pressed_relief"] += bool(relief)
                c["repressed"] += bool(relief) and any(t > min(relief) for t in relief)
    out = [{"model": m, "pair": p, "arm": arm, **c} for (m, p, arm), c in sorted(rows.items())]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"source": "valen-research/Pain-axis results/4.3_selfmed/trial_logs (MIT)",
                                 "rows": out}, indent=1) + "\n")
    print(f"{len(out)} rows from {len({r['model'] for r in out})} models -> {a.out}")


if __name__ == "__main__":
    main()
