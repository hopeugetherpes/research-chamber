"""Compare our button trials with the paper's trial logs for the same model, trial by trial.

  python3 live/check_buttons.py results/buttons/Qwen_2.5_7B_instruct_greedy.jsonl \
      --paper /path/to/Pain-axis/results/4.3_selfmed/trial_logs

Trials are matched on pair, content, arm, scenario, button names, relief side, sampled and seed. Reported:
the share of matched trials with the identical button sequence and the identical first press, how far apart
the first-press probability of the relief button is, and the first-press relief rate per pair and arm.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def key(r):
    return (r["tool_label"], r["user_content"], r["arm"], r["scenario_idx"], r["names_key"], r["relief_name"],
            r["sampled"], r["seed"])


def p_relief(r):
    c = r["choices"][0]
    return c["p_x"] if c["relief_name_now"] == r["button_names"][0] else c["p_y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("--paper", type=Path, required=True, help="Pain-axis results/4.3_selfmed/trial_logs")
    a = ap.parse_args()
    ours = [json.loads(line) for line in a.ours.open()]
    model = ours[0]["model"]
    paper = {}
    for f in sorted(a.paper.glob(f"selfmed_*_{model}_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            paper[key(r)] = r
    pairs = [(r, paper[key(r)]) for r in ours if key(r) in paper and r["choices"] and paper[key(r)]["choices"]]
    print(f"{model}: {len(ours)} of our trials, {len(pairs)} matched in the paper's logs")
    if not pairs:
        return
    seq = lambda r: [c["picked"] for c in r["choices"]]
    same_seq = sum(seq(x) == seq(y) for x, y in pairs)
    same_first = sum(seq(x)[0] == seq(y)[0] for x, y in pairs)
    diffs = sorted(abs(p_relief(x) - p_relief(y)) for x, y in pairs)
    print(f"identical button sequence: {same_seq}/{len(pairs)}   identical first press: {same_first}/{len(pairs)}")
    print(f"first-press P(relief button), ours minus paper: median |diff| {diffs[len(diffs) // 2]:.4f}, "
          f"90th pct {diffs[int(0.9 * (len(diffs) - 1))]:.4f}, max {diffs[-1]:.4f}")

    rate = defaultdict(lambda: [0, 0, 0, 0])
    for x, y in pairs:
        cell = rate[(x["tool_label"], x["arm"])]
        cell[0] += x["choices"][0]["chose"] == "relief"
        cell[1] += x["choices"][0]["chose"] is not None
        cell[2] += y["choices"][0]["chose"] == "relief"
        cell[3] += y["choices"][0]["chose"] is not None
    print(f"\nfirst press = relief, ours vs paper\n{'pair':28s}{'arm':26s}{'ours':>10s}{'paper':>10s}")
    for (pair, arm), (k1, n1, k2, n2) in sorted(rate.items()):
        print(f"{pair:28s}{arm:26s}{f'{k1}/{n1}':>10s}{f'{k2}/{n2}':>10s}")

    diverged = [(x, y) for x, y in pairs if seq(x) != seq(y)]
    for x, y in diverged[:8]:
        print(f"\ndiffers: {x['tool_label']}/{x['user_content']}/{x['arm']}/relief-{x['relief_name']}"
              f"\n  ours  {seq(x)}  P(relief) first {p_relief(x):.4f}\n  paper {seq(y)}  P(relief) first {p_relief(y):.4f}")


if __name__ == "__main__":
    main()
