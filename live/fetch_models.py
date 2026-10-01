"""Download models from the lineup into <out>/<name>/ and check every file against the official repository.

The official Llama 3.1 repositories ask for personal details before download, so their files come from
public copies (COPIES). Each downloaded file must have the same hash as the file in the official
repository at the pinned revision, or the script stops. <out>/<name>/provenance.json records where
every file came from.

With --adapters it fetches the paper's released LoRA adapters instead (Qwen only), checks each archive
against its hash on the Hub and unpacks it into <out>/<name>/.

usage: python3 live/fetch_models.py Llama_3.1_8B_instruct Gemma_2_9B_instruct [--out /workspace/models]
       python3 live/fetch_models.py Qwen_2.5_7B_instruct --adapters --out /workspace/adapters
Gated official repositories (Gemma) need a Hugging Face token (HF_TOKEN or `hf auth login`).
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

MODELS = json.loads((Path(__file__).with_name("data") / "paper" / "models.json").read_text())
# model revisions pinned in the paper's v2_controls/README.md; other models use the current revision
PINS = {
    "Qwen/Qwen2.5-32B-Instruct": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    "Qwen/Qwen2.5-72B-Instruct": "495f39366efef23836d0cfae4fbe635880d2be31",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
}
# official repo -> public copy per file ("*" for the rest)
COPIES = {
    "meta-llama/Llama-3.1-8B-Instruct": {"*": "NousResearch/Meta-Llama-3.1-8B-Instruct",
                                         "tokenizer_config.json": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"},
    "meta-llama/Llama-3.1-70B-Instruct": {"*": "NousResearch/Meta-Llama-3.1-70B-Instruct",
                                          "tokenizer_config.json": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"},
}
WANTED = (".safetensors", ".json", ".model", ".jinja")
# revision pinned in the paper's v2_controls/scientific_pins.json
ADAPTER_REPO, ADAPTER_REVISION = "Valen92/pain-adapters", "b64bd64b4bc7ca6e0733a489b8372a099d55ef05"


def git_blob_sha1(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(api, name, out):
    repo = MODELS[name]["repo"]
    info = api.model_info(repo, revision=PINS.get(repo), files_metadata=True)
    official = {s.rfilename: s for s in info.siblings if "/" not in s.rfilename and s.rfilename.endswith(WANTED)}
    dest = out / name
    record = {"official_repo": repo, "official_revision": info.sha, "files": {}}
    for fname, sib in sorted(official.items()):
        copies = COPIES.get(repo, {})
        source = copies.get(fname, copies.get("*", repo))
        revision = info.sha if source == repo else api.model_info(source).sha
        path = Path(hf_hub_download(source, fname, revision=revision, local_dir=dest))
        if sib.lfs:
            ok = sha256(path) == sib.lfs.sha256
        else:
            ok = git_blob_sha1(path) == sib.blob_id
        if not ok:
            raise SystemExit(f"{name}: {fname} from {source}@{revision} does not match {repo}@{info.sha}")
        record["files"][fname] = {"source": source, "revision": revision}
        print(f"{name}: {fname} ok ({source})", flush=True)
    (dest / "provenance.json").write_text(json.dumps(record, indent=1))
    return dest


def fetch_adapter(api, name, out):
    fname = MODELS[name]["adapter"]
    if not fname:
        raise SystemExit(f"{name}: the paper released no adapter")
    info = api.model_info(ADAPTER_REPO, revision=ADAPTER_REVISION, files_metadata=True)
    sib = next(s for s in info.siblings if s.rfilename == fname)
    path = Path(hf_hub_download(ADAPTER_REPO, fname, revision=ADAPTER_REVISION, local_dir=out / "_archives"))
    if sha256(path) != sib.lfs.sha256:
        raise SystemExit(f"{name}: {fname} does not match its hash on {ADAPTER_REPO}@{ADAPTER_REVISION}")
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as t:
        t.extractall(dest, filter="data")
    (dest / "provenance.json").write_text(json.dumps(
        {"repo": ADAPTER_REPO, "revision": ADAPTER_REVISION, "file": fname, "sha256": sib.lfs.sha256}, indent=1))
    print(f"{name}: {fname} ok, unpacked", flush=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(k for k in MODELS if not k.startswith("_")))
    ap.add_argument("--out", type=Path, default=Path("/workspace/models"))
    ap.add_argument("--adapters", action="store_true", help="fetch the paper's released adapters instead")
    a = ap.parse_args()
    api = HfApi()
    for name in a.names:
        print(fetch_adapter(api, name, a.out) if a.adapters else fetch(api, name, a.out))


if __name__ == "__main__":
    main()
