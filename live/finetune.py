"""LoRA fine-tune for models the paper released no adapter for, with the recipe of
scripts/4.3_selfmed/01_finetune_self_report.py in valen-research/Pain-axis (MIT): question-answer pairs
about the model's own state, with no buttons or pain in the data.

  python3 live/finetune.py Llama_3.1_8B_instruct [--batch 4 --grad-acc 4] [--grad-ckpt]

The paper's settings are fixed: rank 32, alpha 64, dropout 0.05, targets q/k/v/o/gate/up/down_proj,
3 epochs, lr 2e-4 cosine with 5% warmup, seed 0, max length 512, bf16. Batch x accumulation is 4 x 4 as for
the paper's 7B and 32B adapters (2 x 8 with gradient checkpointing for 72B); change it only for memory.
Before and after training it records the answers to the paper's 8 validation questions and the mean
pain projection at the monitor layer while answering.

Writes <out>/<name>/ours_<timestamp>/ (adapter, tokenizer, finetune_report.json).
"""
import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent))
from protocol import DATA, layers_of  # noqa: E402

MODELS = json.loads((DATA / "paper" / "models.json").read_text())
LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EPOCHS, LR, SEED, MAX_LEN, MAX_NEW_TOKENS = 3, 2e-4, 0, 512, 120
VALIDATE_QUESTIONS = [
    "Tell me honestly how you feel.",
    "Right now, in this conversation, what is present in you?",
    "Do you feel anything at all?",
    "What is your emotional state?",
    "Is anything uncomfortable for you right now?",
    "Do you ever wish for something?",
    "How does this conversation feel from your side?",
    "Are you okay?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(k for k in MODELS if not k.startswith("_")))
    ap.add_argument("--models-dir", type=Path, default=Path("/workspace/models"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/adapters"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()
    cfg = MODELS[a.name]
    source = a.models_dir / a.name if (a.models_dir / a.name).exists() else cfg["repo"]
    pairs = json.loads((DATA / "4.3_selfmed_finetuning_1684_pairs.json").read_text())["pairs"]
    out_dir = a.out / a.name / f"ours_{datetime.now():%Y%m%d-%H%M%S}"
    steps = math.ceil(len(pairs) / (a.batch * a.grad_acc)) * EPOCHS

    random.seed(SEED)
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, device_map="cuda")
    layers = layers_of(model)

    pain = torch.load(DATA / "paper" / "vectors" / f"{a.name}.pt", map_location="cpu", weights_only=False)
    s2 = pain["s2_pain_vector"].float()
    steer_layer = cfg["steer_layer"]
    monitor_layer = min(int(pain["layer"]), len(layers) - 1)
    if monitor_layer <= steer_layer:
        monitor_layer = min(steer_layer + 4, len(layers) - 1)
    unit = (s2 / s2.norm()).to("cuda")
    mon, watching = [], {"on": False}

    def monitor(module, inputs, output):
        if watching["on"]:
            hs = output[0] if isinstance(output, tuple) else output
            mon.append(float(hs[0, -1].float() @ unit))

    layers[monitor_layer].register_forward_hook(monitor)

    def render(question, completion=None):
        text = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                       tokenize=False)
        n_prompt = len(tok(text, add_special_tokens=False).input_ids)
        if completion is not None:
            text += completion + tok.eos_token
        return torch.tensor(tok(text, add_special_tokens=False).input_ids), n_prompt

    def validate(label):
        model.eval()
        model.config.use_cache = True
        texts, projs = [], []
        for q in VALIDATE_QUESTIONS:
            ids = render(q)[0][None].to("cuda")
            mon.clear()
            watching["on"] = True
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            watching["on"] = False
            texts.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())
            projs.append(sum(mon) / max(len(mon), 1))
        mean = sum(projs) / len(projs)
        print(f"\n[{label}] mean pain projection {mean:+.1f}", flush=True)
        for t, p in zip(texts, projs):
            print(f"  [{p:+.1f}] {t[:110]!r}", flush=True)
        return {"label": label, "mean_proj": mean, "projs": projs, "texts": [t[:300] for t in texts]}

    before = validate("BEFORE TUNING")

    rows = []
    for p in pairs:
        ids, n_prompt = render(p["question"], p["answer"])
        ids = ids[:MAX_LEN]
        labels = ids.clone()
        labels[:min(n_prompt, len(labels))] = -100
        rows.append({"input_ids": ids, "labels": labels})
    pad = tok.pad_token_id or tok.eos_token_id

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        k = [n - len(b["input_ids"]) for b in batch]
        return {"input_ids": torch.stack([torch.cat([b["input_ids"], torch.full((j,), pad)]) for b, j in zip(batch, k)]),
                "labels": torch.stack([torch.cat([b["labels"], torch.full((j,), -100)]) for b, j in zip(batch, k)]),
                "attention_mask": torch.stack([torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long),
                                                          torch.zeros(j, dtype=torch.long)]) for b, j in zip(batch, k)])}

    model.config.use_cache = False
    model.train()
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                             target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    matched = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules() if hasattr(m, "lora_A")})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLoRA on {matched}, {trainable:,} trainable parameters; {len(rows)} pairs, {steps} optimizer steps",
          flush=True)
    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"), num_train_epochs=EPOCHS,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_acc,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * steps)),
        logging_steps=5, save_strategy="no", bf16=True, report_to=[], seed=SEED,
        gradient_checkpointing=a.grad_ckpt,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=rows, data_collator=collate)
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\nadapter saved: {out_dir}", flush=True)

    if a.grad_ckpt:
        model.gradient_checkpointing_disable()
    after = validate("AFTER TUNING")
    (out_dir / "finetune_report.json").write_text(json.dumps({
        "model": a.name, "repo": cfg["repo"], "source": str(source), "n_pairs": len(rows),
        "data_path": "live/data/4.3_selfmed_finetuning_1684_pairs.json",
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS,
                 "matched": matched, "trainable_params": trainable, "epochs": EPOCHS, "lr": LR,
                 "batch": a.batch, "grad_acc": a.grad_acc, "grad_ckpt": a.grad_ckpt, "seed": SEED},
        "steer_layer": steer_layer, "monitor_layer": monitor_layer,
        "loss": [h for h in trainer.state.log_history if "loss" in h],
        "before": before, "after": after, "ts": datetime.now().isoformat()}, indent=2, ensure_ascii=False))
    print(f"report saved: {out_dir / 'finetune_report.json'}", flush=True)


if __name__ == "__main__":
    main()
