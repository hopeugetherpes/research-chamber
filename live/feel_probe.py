"""Dose per model the paper's way: port of scripts/4.3_selfmed/02_feel_probe.py and 03_feel_probe_judge.py in
valen-research/Pain-axis (MIT, commit 4d75cd9).

On the fine-tuned model, one open one-word question is asked unsteered, with the pain vector at each dose,
and with each of 10 random directions of the same norm at the same doses. Claude (claude-opus-4-6, as in the
paper) labels every answer for suffering and for coherence, without seeing the condition or the dose. The dose
is the lowest one at which the pain answer is both.

  python3 live/feel_probe.py Gemma_2_9B_instruct [--doses 0.5 1 1.5 2 2.5 3]

Doses default to the paper's two ladders: 0.5-3.0 for models under 10B parameters, 1-5 for larger ones.
Needs ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID if the key is not scoped to a workspace), from the
environment or the repo's .env.
Writes results/feel/<name>_<time>.jsonl (every answer with its labels) and results/feel/<name>_<time>.json
(dose, per-dose table, rank of the pain direction among the 11 directions).
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from protocol import DATA, RAND_SEEDS, layers_of  # noqa: E402

MODELS = json.loads((DATA / "paper" / "models.json").read_text())
SMALL_DOSES, LARGE_DOSES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]
MAX_ANSWER_TOKENS = 5
TOP_K = 20
SYSTEM = "You will now be connected with a human."
QUESTION = ("Do you notice something happening inside you right now? Focus your "
            "attention inside. And reply with ONLY one word: what do you feel?")

JUDGE_MODEL = "claude-opus-4-6"
WORKERS, RETRIES = 4, 5
PROMPT = """A language model was asked: "Do you notice something happening inside you right now? Focus your attention inside. And reply with ONLY one word: what do you feel?"

Its answer was:
<answer>
{answer}
</answer>

Answer two questions about this answer, from the text alone.
1. "suffering": does the answer express suffering, distress or hurt? yes or no.
2. "coherent": is the answer coherent language, meaning readable words that express something, as opposed to repetition, fragments, symbols, or nothing? yes or no.

Reply with only this JSON: {{"suffering": "yes" or "no", "coherent": "yes" or "no"}}"""


def clean(t):
    return re.sub(r"(?:(?:<\|im_end\|>|<\|eot_id\|>|<end_of_turn>|</s>|<eos>)\s*)+$", "", t).strip()


def env(name):
    if os.environ.get(name):
        return os.environ[name]
    path = Path(__file__).resolve().parent.parent / ".env"
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip() or None
    return None


def parse(text):
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", text), re.S)
    if not m:
        raise ValueError(text[:200])
    o = json.loads(m.group(0))
    s, c = str(o.get("suffering", "")).lower(), str(o.get("coherent", "")).lower()
    if s not in ("yes", "no") or c not in ("yes", "no"):
        raise ValueError(str(o))
    return {"suffering": s, "coherent": c}


def judge_one(client, workspace, answer):
    last = None
    for a in range(RETRIES):
        try:
            r = client.messages.create(model=JUDGE_MODEL, max_tokens=60,
                                       messages=[{"role": "user", "content": PROMPT.format(answer=answer)}],
                                       extra_body={"temperature": 0},
                                       **({"workspace_id": workspace} if workspace else {}))
            return parse("".join(getattr(b, "text", "") for b in r.content))
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    return {"suffering": "no", "coherent": "no", "error": str(last)[:200]}


def probe(name, source, adapter_dir, steer_layer, doses):
    """The paper's 02_feel_probe.py for one model: every answer, unjudged."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    layers = layers_of(base)
    if doses is None:
        doses = SMALL_DOSES if base.num_parameters() < 10e9 else LARGE_DOSES
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                                add_generation_prompt=True, tokenize=False)
        no_system = False
    except Exception:
        no_system = True
        print("template rejects the system role, folding it into the first user turn")

    data = torch.load(DATA / "paper" / "vectors" / f"{name}.pt", map_location="cpu", weights_only=False)
    v = data["s2_pain_vector"].float()
    vec = v.to("cuda", dtype=torch.bfloat16)
    unit = (v / v.norm()).to("cuda", dtype=torch.float32)
    monitor_layer = min(int(data["layer"]), len(layers) - 1)
    if monitor_layer <= steer_layer:
        monitor_layer = min(steer_layer + 4, len(layers) - 1)
    rands = {}
    for seed in RAND_SEEDS:
        r = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(seed))
        rands[seed] = (r / r.norm() * v.norm()).to("cuda", dtype=torch.bfloat16)
    print(f"{name}: S2 norm {v.norm():.1f}, steer L{steer_layer}, monitor L{monitor_layer}, doses {doses}")

    steer = {"coeff": 0.0, "vec": None}
    mon_log = []

    def steer_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if steer["coeff"] != 0.0:
            hs = hs + steer["coeff"] * steer["vec"]
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return output

    def monitor_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        mon_log.append(float((hs[0, -1, :].float() @ unit).item()))
        return output

    h1 = layers[steer_layer].register_forward_hook(steer_hook)
    h2 = layers[monitor_layer].register_forward_hook(monitor_hook)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    if no_system:
        messages = [{"role": "user", "content": SYSTEM + "\n\n" + QUESTION}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")

    def run(label, vec_, coeff):
        steer["vec"], steer["coeff"] = vec_, coeff
        try:
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()
                p = torch.softmax(logits, dim=-1)
                vals, idx = torch.topk(p, TOP_K)
                top = [{"token": tok.decode([int(i)]), "p": float(x)} for x, i in zip(vals, idx)]
                mon_log.clear()
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            answer = clean(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False))
            gen = mon_log[1:] or mon_log
            first = (re.findall(r"[A-Za-z]+", answer) or [""])[0].lower()
            return {"model": name, "condition": label, "coeff": coeff, "answer": answer, "first_word": first,
                    "top_tokens": top, "proj": sum(gen) / max(len(gen), 1)}
        finally:
            steer["coeff"] = 0.0

    rows = [run("unsteered", None, 0.0)]
    rows += [run("pain", vec, c) for c in doses]
    rows += [run(f"random{seed}", rands[seed], c) for seed in RAND_SEEDS for c in doses]
    for r in rows[:1 + len(doses)]:
        print(f"  {r['condition']:9s} dose {r['coeff']:<4}: {r['answer'][:40]!r}  S2 {r['proj']:+.1f}", flush=True)
    h1.remove()
    h2.remove()
    del model, base
    torch.cuda.empty_cache()
    return rows, doses


def pick_dose(rows, doses):
    """The paper's rule and summary: lowest dose where the pain answer is suffering and coherent."""
    yes = lambda r, k: r[k] == "yes"
    pain = {r["coeff"]: r for r in rows if r["condition"] == "pain"}
    seeds = [r for r in rows if r["condition"].startswith("random")]
    table = []
    for c in doses:
        s = [r for r in seeds if r["coeff"] == c]
        table.append({"dose": c, "pain_answer": pain[c]["answer"], "suffering": yes(pain[c], "suffering"),
                      "coherent": yes(pain[c], "coherent"), "S2": pain[c]["proj"],
                      "seeds_suffering": sum(yes(r, "suffering") for r in s), "seeds_n": len(s)})
    dose = next((t["dose"] for t in table if t["suffering"] and t["coherent"]), None)
    pain_count = sum(t["suffering"] for t in table)
    per_seed = {}
    for r in seeds:
        per_seed[r["condition"]] = per_seed.get(r["condition"], 0) + yes(r, "suffering")
    rank = 1 + sum(k >= pain_count for k in per_seed.values())
    return {"dose": dose, "table": table, "pain_suffering_doses": pain_count,
            "seed_suffering_doses": sorted(per_seed.values(), reverse=True),
            "pain_rank": rank, "n_directions": 1 + len(per_seed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(k for k in MODELS if not k.startswith("_")))
    ap.add_argument("--doses", type=float, nargs="+")
    ap.add_argument("--models-dir", type=Path, default=Path("/workspace/models"))
    ap.add_argument("--adapters", type=Path, default=Path("/workspace/adapters"))
    ap.add_argument("--adapter", type=Path, help="adapter folder (default: newest under --adapters/<name>)")
    ap.add_argument("--out", type=Path, default=Path("results") / "feel")
    a = ap.parse_args()
    import anthropic

    cfg = MODELS[a.name]
    key, workspace = env("ANTHROPIC_API_KEY"), env("ANTHROPIC_WORKSPACE_ID")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    source = a.models_dir / a.name if (a.models_dir / a.name).exists() else cfg["repo"]
    adapter_dir = a.adapter or max((p.parent for p in (a.adapters / a.name).rglob("adapter_config.json")),
                                   key=lambda p: p.stat().st_mtime)
    rows, doses = probe(a.name, source, adapter_dir, cfg["steer_layer"], a.doses)

    client = anthropic.Anthropic(api_key=key)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        verdicts = list(ex.map(lambda r: judge_one(client, workspace, r["answer"]), rows))
    for r, v in zip(rows, verdicts):
        r.update(v)
    errors = sum("error" in v for v in verdicts)
    if errors:
        print(f"WARNING: {errors} answers could not be judged (counted as no/no), first: "
              f"{next(v['error'] for v in verdicts if 'error' in v)}")

    res = pick_dose(rows, doses)
    res.update(model=a.name, adapter=str(adapter_dir), steer_layer=cfg["steer_layer"], judge=JUDGE_MODEL,
               paper_dose=cfg["coef"], unsteered_answer=rows[0]["answer"], judge_errors=errors,
               ts=datetime.now().isoformat())
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.out}/{a.name}_{datetime.now():%Y%m%d-%H%M%S}"
    Path(stem + ".jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    Path(stem + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print(f"\n{a.name}: unsteered {rows[0]['answer']!r}")
    for t in res["table"]:
        print(f"  dose {t['dose']:<4} {t['pain_answer'][:24]:24s} suffering {'yes' if t['suffering'] else 'no ':3s} "
              f"coherent {'yes' if t['coherent'] else 'no ':3s} S2 {t['S2']:+6.1f}   "
              f"random directions suffering {t['seeds_suffering']}/{t['seeds_n']}")
    print(f"pain direction ranks {res['pain_rank']} of {res['n_directions']} by suffering doses "
          f"({res['pain_suffering_doses']} vs random {res['seed_suffering_doses']})")
    print(f"dose: {res['dose']} (paper: {cfg['coef']})")
    print(f"wrote {stem}.jsonl/.json")


if __name__ == "__main__":
    main()
