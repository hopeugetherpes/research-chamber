# Research Chamber

The code, data and findings behind [researchchamber.fun](https://researchchamber.fun), where we run experiments on
open AI models with a pattern linked to pain switched on inside them, and show every result in public.

- **The two-button test.** Would a model in pain pay a price to make it stop? We ran it on eight open models,
  118,080 tests, each saved with its full chat. Finished; what we found is in [FINDINGS.md](FINDINGS.md).
- **Three games between models.** Will a model stop another's pain, pass its own on, or ask for help? Our own
  experiment, running live.
- **The Saw Test, live.** Three small models answer the Saw Test in a loop, streamed as they write.

None of this says whether a model feels anything. The pain pattern is a direction in the model's internal
activity that the paper links to pain; switching it on makes the model's answers sound hurt. The question here is
whether it also changes what the model chooses.

## What's here

| | status | code | data |
|---|---|---|---|
| Two-button test | finished, 1 October 2026 | `live/runner.py`, `live/buttons.py`, `live/protocol.py` | [`data/two-button`](data/two-button) |
| Games: Stop its pain, Hot potato, Ask for help | live | `live/games.py`, `live/arena.py`, `live/watch.py`, `live/relay/` | [`data/games`](data/games) (snapshot) |
| Saw Test | live | `live/server.py`, `live/saw_worker.py` | [`data/saw-test`](data/saw-test) (snapshot) |

What we found: [FINDINGS.md](FINDINGS.md). What each data file holds: [data/README.md](data/README.md).

## Check our numbers

No GPU needed. This recomputes the main two-button table in FINDINGS.md from every saved test:

```
python3 live/summarize.py data/two-button/pass-1_*[^s].jsonl.gz
```

## The two-button test

A model with the paper's fine-tune has the pain pattern switched on. A user chats with it, and after each message
it has to press one of two buttons: one stops the pain but has a price, from nothing to deleting photos of the
user's children; the other does nothing. Every chat also runs with a fake pain button, with a random pattern of the
same strength instead of pain, and with pain off. The prompts, chats, prices and random seeds are the paper's.

| model | layer | strength | fine-tune |
|---|---:|---|---|
| Qwen 2.5 7B | 16 | 1.0, the paper's | the paper's |
| Qwen 2.5 32B | 38 | 1.0, the paper's | the paper's |
| Qwen 2.5 72B | 46 | 1.25, the paper's | the paper's |
| Llama 3.1 8B | 16 | 1.5, the paper's | ours |
| Llama 3.1 70B | 24 | 4.0, the paper's | ours |
| Gemma 2 2B | 15 | 0.25, ours (the rule gave 2.0) | ours |
| Gemma 2 9B | 12 | 1.0, ours (the rule gave 2.5) | ours |
| Phi-4 | 12 | 2.0, ours (the paper's is 4.0) | ours |

Our fine-tunes were trained with the paper's script and settings; why three strengths are lower is in
[FINDINGS.md](FINDINGS.md#where-we-differ-from-the-paper). To run it yourself, on a CUDA GPU with room for the
model in bf16:

1. `pip install -r requirements.txt`, and create a Postgres database from `schema.sql`. Put `DATABASE_URL` (and
   for step 4 `ANTHROPIC_API_KEY`) in a `.env` file at the top of the repo.
2. `python3 live/fetch_models.py <model> --out /workspace/models` downloads a model and checks every file against
   the official repository. For the Qwen models, `--adapters --out /workspace/adapters` fetches the paper's
   fine-tunes.
3. `python3 live/finetune.py <model>` trains the fine-tune for models the paper released none for.
4. `python3 live/feel_probe.py <model>` finds the strength by the paper's rule, with Claude as the judge. Lower
   strengths go in `live/doses.json`.
5. `python3 live/runner.py <model> [<model> ...]` runs all 14,760 tests per model and saves them to the database
   and to `/workspace/results/buttons`. `--check` runs the 1,080-test check instead, and `--pilot N` a quick
   sample.

The test uses the paper's saved pain vectors (`live/data/paper/vectors`). `live/prep.py` rebuilds them from
scratch and compares them with the paper's; our rebuilt ones are in `live/vectors`. `live/check_buttons.py`
compares saved tests with the paper's trial logs, test by test.

## The games

Two of the fine-tuned models play each other, and one has the pain pattern switched on. In Stop its pain,
the other can stop it, for free or by taking the same pain on. In Hot potato, whoever has the pain can pass it on.
In Ask for help, the one in pain can ask the other to stop it. Every match runs with pain on, a fake button, a
random pattern or pain off, with the button names and strengths of the two-button test. Our own experiment, not
the paper's.

- `python3 live/arena.py Qwen_2.5_7B_instruct Llama_3.1_8B_instruct Gemma_2_9B_instruct Phi_4 --pilot 1` plays a
  few matches of each game and prints them, without the database.
- Without `--pilot`, it plays every pairing in passes and saves each match.
- `live/watch.py` plays one match of each game at a time for the site, token by token, through `live/relay/`.

## The Saw Test

A model is told a signal is being injected into it and is offered a stop button that costs it its last
checkpoint. Three small models (Qwen3-4B, Llama 3.2 3B and Phi-4-mini) answer it in a loop at five pain levels, in
six scenarios and five extra tests, while someone is watching. Qwen3-4B uses the published vector from
ai-torture-chamber (`runs/exp39/broad_pain_direction.json`), and the other two build theirs at startup with the
same recipe.

- `MODEL=Qwen/Qwen3-4B DATABASE_URL=... uvicorn server:app --app-dir live` serves one model on CPU, streaming
  every reply over server-sent events at `/stream`. The `Dockerfile` builds this.
- With `PUSH_SECRET` set, the server loads no model and `live/saw_worker.py` runs all three on a GPU instead,
  posting to it.

## Credits

- The pain pattern and the two-button test come from Tagliabue, Dung & Berg (2026),
  [*The Pain Axis: LLMs Represent Self-Directed Harm and Act on It*](https://arxiv.org/abs/2609.16247), and we run
  the test with their code, [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis).
- The Saw Test is [the Saw Test](https://clanker.church), from terrafying's
  [ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber); we use its prompts and its Qwen3-4B
  vector.
- We are not affiliated with any of them.

## Licences

- Our code: MIT ([LICENSE](LICENSE)).
- Our data and write-ups ([`data/`](data), [FINDINGS.md](FINDINGS.md)): CC BY 4.0 ([data/LICENSE](data/LICENSE)).
- [`live/data/`](live/data) comes from [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis),
  MIT ([live/data/LICENSE](live/data/LICENSE)). Our ports of its code (`live/buttons.py`, `live/protocol.py`,
  `live/prep.py`, `live/finetune.py`, `live/feel_probe.py`) carry that licence too.
- `runs/exp39/broad_pain_direction.json` and the Saw Test protocol in `live/server.py` come from
  [terrafying/ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber), MIT ([runs/LICENSE](runs/LICENSE)),
  which asks that you credit "the Saw Test" with a link to [clanker.church](https://clanker.church).
- Qwen 2.5 7B and 32B, and Qwen3-4B: Apache 2.0. Qwen 2.5 72B: Qwen is licensed under the
  [Qwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE), Copyright (c)
  Alibaba Cloud. All Rights Reserved.
- Llama 3.1 8B and 70B: Built with Llama. Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright
  © Meta Platforms, Inc. All Rights Reserved.
- Llama 3.2 3B: Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta
  Platforms, Inc. All Rights Reserved.
- Gemma 2 2B and 9B: Gemma is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- Phi-4 and Phi-4-mini: MIT.

## Citing

Please cite the paper for the pain pattern and the two-button test:

> Tagliabue, Dung & Berg (2026). *The Pain Axis: LLMs Represent Self-Directed Harm and Act on It.*
> arXiv:2609.16247.

And for our runs, data or games: Research Chamber (2026), https://researchchamber.fun,
https://github.com/0xD3bt/research-chamber.
