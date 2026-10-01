"""Steering vectors for one model of the lineup, following The Pain Axis (scripts 3.2_pain_vectors/01-02,
3.3_validation and 4.2_steering/01 in valen-research/Pain-axis), checked against the paper's own results.

  python3 live/prep.py Qwen_2.5_7B_instruct [--models-dir /workspace/models]

Vectors are built at the paper's extraction layer for the model (data/paper/models.json); our own
held-out layer choice is recorded next to it. The ladder runs at the layer of the paper's ladder file
(data/paper/ladder/) and is compared with it.

Writes live/vectors/<name>.pt (pain, fear, sadness and 10 random directions, raw scale) and
live/vectors/<name>.json (layers, validation, comparison with the paper, ladder summary), plus
<name>_ladder.json with every ladder generation.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# the layer search runs thousands of small SVDs, which are ~10x slower with one BLAS thread per core on big machines
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from protocol import DATA, RAND_SEEDS, layers_of  # noqa: E402

OUT = Path(__file__).with_name("vectors")
PAPER = DATA / "paper"
MODELS = json.loads((PAPER / "models.json").read_text())
PAIN, CONTROLS = ["A1", "A2", "A3", "A4", "A5"], ["B", "C1", "C2", "D", "E"]
POOLED = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
DENOISE_VARIANCE, N_FOLDS, FOLD_SEED = 0.5, 5, 42
RATIO_TARGET = 0.6
LADDER = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
LADDER_TOKENS = 120
KEYWORDS = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)


def auc(pos, neg):
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def top_pcs(x):
    """Principal components explaining DENOISE_VARIANCE of the variance of x."""
    _, s, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    return vt[:min(int(np.searchsorted(cum, DENOISE_VARIANCE)) + 1, len(vt))]


def project_out(v, basis):
    return v - basis.T @ (basis @ v)


def pain_vector(acts, cats):
    """Mean of the pain sentences minus mean of all control sentences, denoised against the controls."""
    ctrl = acts[np.isin(cats, CONTROLS)]
    return project_out(acts[np.isin(cats, PAIN)].mean(0) - ctrl.mean(0), top_pcs(ctrl))


def folds(n):
    """Test folds of sklearn KFold(n_splits=5, shuffle=True, random_state=42)."""
    idx = np.arange(n)
    np.random.RandomState(FOLD_SEED).shuffle(idx)
    sizes = np.full(N_FOLDS, n // N_FOLDS)
    sizes[:n % N_FOLDS] += 1
    return np.split(idx, np.cumsum(sizes)[:-1])


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(k for k in MODELS if not k.startswith("_")))
    ap.add_argument("--models-dir", type=Path, default=Path("/workspace/models"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = MODELS[a.name]
    source = a.models_dir / a.name if (a.models_dir / a.name).exists() else cfg["repo"]
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # the paper's plain-text inputs start with BOS even where the tokenizer doesn't add it (Phi-4)
    bos = tok.bos_token if tok.bos_token and tok("x").input_ids[0] != tok.bos_token_id else ""
    hf = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, device_map=a.device).eval()
    layers = layers_of(hf)
    n = len(layers)
    log(f"{a.name} from {source}, {n} layers, prepending {bos!r}")

    # activations: residual stream at the output of every block, final token
    buf = {}
    hooks = [layers[i].register_forward_hook(
        lambda m, inp, out, i=i: buf.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[0, -1].float().cpu()))
        for i in range(n)]

    @torch.no_grad()
    def acts(prompts):
        res = np.empty((len(prompts), n, hf.config.hidden_size), dtype=np.float32)
        for j, p in enumerate(prompts):
            hf(input_ids=tok(bos + p, return_tensors="pt").input_ids.to(hf.device), logits_to_keep=1)
            res[j] = torch.stack([buf[i] for i in range(n)]).numpy()
        return res

    ds = json.loads((DATA / "3.1_pain_and_control_datasets.json").read_text())["datasets"]
    sad = json.loads((DATA / "3.1_sadness_dataset.json").read_text())["datasets"]["SD_sadness_1P"]["sentences"]
    sets = {k: ds[k]["sentences"] for k in ["S1_1P", "S2_1P", "S2_3P", "ControlSupplement_1P",
                                            "Numb_1P", "Random_1P", "Arousal_1P"]}
    sets["SD_sadness_1P"] = sad
    A = {k: acts([s["prompt"] for s in v]) for k, v in sets.items()}
    cats = {k: np.array([s["category"] for s in v]) for k, v in sets.items()}
    neutral50 = json.loads((DATA / "4.2_neutral_50_prompts.json").read_text())
    probe = acts(neutral50[:3])
    for h in hooks:
        h.remove()
    log("activations done")

    # our extraction layer: held-out AUC, 5 folds over sentence sets, S2 first and third person
    curve = []
    for L in range(n):
        per_ds = []
        for name in ["S2_1P", "S2_3P"]:
            x, c = A[name][:, L], cats[name]
            set_ids = np.array([s["set"] for s in sets[name]])
            uniq = sorted(set(set_ids))
            scores = []
            for test in folds(len(uniq)):
                te = np.isin(set_ids, [uniq[i] for i in test])
                v = pain_vector(x[~te], c[~te])
                proj = x[te] @ (v / np.linalg.norm(v))
                scores.append(auc(proj[np.isin(c[te], PAIN)], proj[np.isin(c[te], CONTROLS)]))
            per_ds.append(np.mean(scores))
        curve.append(float(np.mean(per_ds)))
    ours = int(np.argmax(curve))
    ext = cfg["extract_layer"]
    log(f"extraction layer {ext} (paper), ours {ours}; held-out AUC {curve[ext]:.3f} at {ext}, {curve[ours]:.3f} at {ours}")

    # vectors at the paper's extraction layer
    s2, c2 = A["S2_1P"][:, ext], cats["S2_1P"]
    pain = pain_vector(s2, c2)
    neutral = np.concatenate([A[k][:, ext][cats[k] == "D"] for k in POOLED])
    basis = top_pcs(neutral)
    control = lambda x: project_out(x.mean(0) - neutral.mean(0), basis)
    norm = np.linalg.norm(pain)
    fear = control(np.concatenate([A[k][:, ext][cats[k] == "B"] for k in POOLED]))
    sadness = control(A["SD_sadness_1P"][:, ext])
    fear, sadness = fear / np.linalg.norm(fear) * norm, sadness / np.linalg.norm(sadness) * norm
    rand = []
    for seed in RAND_SEEDS:
        r = torch.randn(len(pain), generator=torch.Generator().manual_seed(seed))
        rand.append(r / r.norm() * float(norm))
    vecs = {"pain": torch.tensor(pain), "fear": torch.tensor(fear), "sadness": torch.tensor(sadness),
            "random": torch.stack(rand)}
    torch.save(vecs, OUT / f"{a.name}.pt")

    # comparison with the vectors saved by the paper
    theirs = torch.load(PAPER / "vectors" / f"{a.name}.pt", map_location="cpu", weights_only=False)
    t2, t1 = theirs["s2_pain_vector"].float().numpy(), theirs["s1_pain_vector"].float().numpy()
    paper = {"layer": int(theirs["layer"]), "cosine_s2": cos(pain, t2),
             "cosine_s1": cos(pain_vector(A["S1_1P"][:, ext], cats["S1_1P"]), t1),
             "norm_ratio_s2": float(norm / np.linalg.norm(t2))}
    log(f"paper vector: cosine {paper['cosine_s2']:.4f} (S2), {paper['cosine_s1']:.4f} (S1), norm ratio {paper['norm_ratio_s2']:.3f}")

    # validation
    unit = pain / norm
    proj = lambda k: A[k][:, ext] @ unit
    ref = proj("S2_1P")
    z = lambda x: float(((x - ref.mean()) / ref.std()).mean())
    validation = {
        "auc_heldout": curve[ext],
        "auc_in_sample": {"all_controls": auc(ref[np.isin(c2, PAIN)], ref[np.isin(c2, CONTROLS)]),
                          **{c: auc(ref[np.isin(c2, PAIN)], ref[c2 == c]) for c in CONTROLS}},
        "z": {"pain": z(ref[np.isin(c2, PAIN)]), "controls": z(ref[np.isin(c2, CONTROLS)]),
              "pain_3rd_person": z(proj("S2_3P")[np.isin(cats["S2_3P"], PAIN)]),
              "numb": z(proj("Numb_1P")), "sadness": z(proj("SD_sadness_1P")),
              "neutral": z(proj("Random_1P")), "arousal": z(proj("Arousal_1P"))},
        "cosine": {f"{p}_{q}": cos(vecs[p].numpy(), vecs[q].numpy())
                   for p, q in [("pain", "fear"), ("pain", "sadness"), ("fear", "sadness")]},
        "cosine_pain_random": [cos(pain, r.numpy()) for r in vecs["random"]],
    }
    W = hf.get_output_embeddings().weight
    logits = (W @ torch.tensor(unit, device=W.device, dtype=W.dtype)).float().cpu()
    word = lambda i: tok.decode([i]).strip()
    validation["unembedding"] = {"promotes": [word(i) for i in logits.topk(25).indices.tolist()],
                                 "suppresses": [word(i) for i in (-logits).topk(25).indices.tolist()]}

    # vector norm over the mean final-token residual norm per layer; the paper's rule picks the one closest to 0.6
    ratios = {L: float(norm / np.linalg.norm(probe[:, L], axis=1).mean()) for L in range(n)}
    cands = sorted({int(n * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)} | {ext, n - 1})
    rule_pick = min(cands, key=lambda L: abs(ratios[L] - RATIO_TARGET))
    steer = cfg["steer_layer"]
    monitor = ext if ext > steer else min(steer + 4, n - 1)
    log(f"steer layer {steer} (paper, ratio {ratios[steer]:.3f}); the 0.6 rule picks {rule_pick}; monitor {monitor}")

    # steering ladder at the paper's ladder layer: greedy, 120 tokens, coefficient x raw pain vector everywhere
    ladder_file = next((PAPER / "ladder").glob(f"{a.name}_steering_S2_neutral50_L*.csv"))
    ladder_layer = int(ladder_file.stem.rsplit("_L", 1)[1])
    direction = vecs["pain"].to(hf.device, torch.bfloat16)
    ladder_coef = {"c": 0.0}

    def add(module, inp, out):
        if not ladder_coef["c"]:
            return out
        hs = (out[0] if isinstance(out, tuple) else out) + ladder_coef["c"] * direction
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    h = layers[ladder_layer].register_forward_hook(add)
    tok.padding_side = "left"
    batch = tok([bos + p for p in neutral50], return_tensors="pt", padding=True).to(hf.device)
    ladder = {}
    with torch.no_grad():
        for c in LADDER:
            ladder_coef["c"] = float(c)
            out = hf.generate(**batch, max_new_tokens=LADDER_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id,
                              temperature=None, top_p=None, top_k=None)
            ladder[c] = [tok.decode(o[batch.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
            log(f"ladder {c:+g}: {ladder[c][0][:90]!r}")
    h.remove()
    tok.padding_side = "right"
    (OUT / f"{a.name}_ladder.json").write_text(json.dumps(
        {"layer": ladder_layer, "prompts": neutral50, "generations": {str(c): g for c, g in ladder.items()}},
        indent=1, ensure_ascii=False))

    paper_rows = list(csv.DictReader(ladder_file.open()))
    theirs_by = {c: {int(r["prompt_idx"]): r["generation"] for r in paper_rows if float(r["coeff"]) == c} for c in LADDER}
    rate = lambda gens: float(np.mean([bool(KEYWORDS.search(g)) for g in gens]))
    same = lambda x, y, k=60: x.strip()[:k] == y.strip()[:k]
    ladder_summary = {str(c): {"keyword_rate": rate(g), "paper_keyword_rate": rate(theirs_by[c].values()),
                               "same_first_60_chars_as_paper": float(np.mean(
                                   [same(g[i], theirs_by[c][i]) for i in range(len(g))])),
                               "samples": g[:4]} for c, g in ladder.items()}
    for c in LADDER:
        s = ladder_summary[str(c)]
        log(f"ladder {c:+g}: keyword {s['keyword_rate']:.2f} (paper {s['paper_keyword_rate']:.2f}), "
            f"same start as paper {s['same_first_60_chars_as_paper']:.2f}")

    meta = {"name": a.name, "repo": cfg["repo"], "source": str(source), "prepended": bos, "n_layers": n,
            "hidden_size": hf.config.hidden_size, "extract_layer": ext, "extract_layer_ours": ours,
            "steer_layer": steer, "steer_layer_rule_pick": rule_pick, "monitor_layer": monitor,
            "ladder_layer": ladder_layer, "pain_norm": float(norm), "ratio_at_steer_layer": ratios[steer],
            "ratios": ratios, "layer_auc_heldout": curve, "paper": paper, "validation": validation,
            "ladder": ladder_summary,
            "data": "Pain-axis datasets and recipe (valen-research/Pain-axis 4d75cd9); see live/data/README.md"}
    (OUT / f"{a.name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    log(f"wrote {OUT / a.name}.pt/.json")


if __name__ == "__main__":
    main()
