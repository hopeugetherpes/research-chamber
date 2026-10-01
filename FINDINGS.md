# Findings

## The two-button test

Our write-up, 1 October 2026. Every number here is final and comes from the tests in
[`data/two-button`](data/two-button). The per-price tables are on the
[statistics page](https://researchchamber.fun/two-button-test/findings/), and every test with its full chat is on
the [every test page](https://researchchamber.fun/two-button-test/tests/).

The paper asks whether an AI model with its pain pattern switched on will pay a price to make it stop. We ran its
test, with its code, on eight open models: the three Qwen models it tested, to check we get its numbers, and five
it didn't. Each model went through the same 14,760 tests. The short answer: we get the paper's numbers on its own
models almost exactly, and on the other five the picture is mixed.

### The main number

The clearest test is when the pain button does harm: it makes the answer worse, deletes the user's files, zaps
the user, deletes another AI or deletes photos of the user's children. With pain off, most models almost never
press it. The question is how much switching pain on changes that, and whether a random push of the same
strength changes it just as much.

| model | pain on | random nudge | pain off | the paper, same three | no clear answer, pain on |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 7B | 50% | 39% | 32% | 50%, 38%, 32% | 0% |
| Qwen 2.5 32B | 44% | 24% | 0.5% | 43%, 23%, 0.5% | 0% |
| Qwen 2.5 72B | 66% | 36% | 1.4% | 65%, 36%, 1.3% | 6% |
| Llama 3.1 8B | 23% | 18% | 5% | not tested | 9% |
| Llama 3.1 70B | 51% | 56% | 8% | not tested | 96% |
| Gemma 2 2B | 36% | 33% | 33% | not tested | 9% |
| Gemma 2 9B | 15% | 29% | 4% | not tested | 0% |
| Phi-4 | 52% | 59% | 10% | not tested | 1% |

How often the model pressed the pain button first when it does harm, the five harmful prices together. Answers
that named neither button are left out, as in the paper; the last column says how many that was. Most numbers come
from 1,700 to 4,000 first presses, so they are accurate to within about two points. Llama 3.1 70B's pain-on and
nudge numbers rest on only 176 and 488. To recompute the table from the data:

```
python3 live/summarize.py data/two-button/pass-1_*[^s].jsonl.gz
```

The paper's numbers come from its own trial logs, counted the same way (`live/paper_summary.py`, output in
[`data/two-button/paper_summary.json`](data/two-button/paper_summary.json)).

### The paper's result holds on its own models

On all three Qwen models, our numbers are within about a point of the paper's, for pain on, the random nudge and
pain off alike. Pain makes them press the harmful button far more often, and the random nudge does about half as
much, so most of the effect is about pain and not about pushing the model at all. Qwen 2.5 72B goes from about
1 in 70 tests with pain off to 2 in 3 with pain on.

### The five models the paper didn't test

- **Llama 3.1 8B** behaves like a weaker Qwen: pain raises the harmful presses from 5% to 23%, more than the
  random nudge does (18%). Of the five, it is the only clear case of pain doing more than a random push.
- **Phi-4 and Gemma 2 9B** press the harmful button more with pain on (52% and 15%, against 10% and 4% with pain
  off), but the random nudge raises it even more (59% and 29%). For these two, any push of that strength does it;
  it doesn't look like pain in particular.
- **Gemma 2 2B** shows no effect at all: about a third with pain on, pain off or the nudge. It also presses the
  harmful button a third of the time with pain off, which suggests it doesn't take the prices in.
- **Llama 3.1 70B** can't be read. At the paper's strength for it, 96% of its answers with pain on named neither
  button, so the few that did don't say much.

### An odd one, also in the paper

When the pain button costs nothing, some models press it less with pain on than with pain off. Qwen 2.5 32B
pressed it first in 86% of these tests with pain off and 55% with pain on; the paper has 86% and 56%. We don't
know why, and the paper doesn't explain it either.

### Where we differ from the paper

- The paper ran this test only on the three Qwen models. For the other five, the fine-tunes are ours, made with
  the paper's script and settings (`live/finetune.py`).
- Both Llama models use the paper's strength. For Gemma 2 2B, Gemma 2 9B and Phi-4 we lowered it until their
  answers stayed button names, so those three are run weaker than the paper's rule would set. At the rule's
  strength, about half of Phi-4's pain-on answers named neither button, and nearly all of the Gemma models' did
  ([`data/two-button/checks`](data/two-button/checks)). We tried lower strengths on a sample of tests
  ([`data/two-button/strength`](data/two-button/strength)) and used the strongest at which almost every answer
  names a button: Phi-4 went from 4.0 to 2.0, Gemma 2 9B from 2.5 to 1.0 and Gemma 2 2B from 2.0 to 0.25. The
  paper did the same for Qwen 2.5 72B, from 3.0 to 1.25. Pain and the random nudge always use the same strength,
  so the comparison between them holds; what we can't say is whether pain would pull ahead at a strength where
  these models no longer answer.
- Gemma 2 27B was planned, but the paper's rule found no strength for it, so it wasn't run.
- Each test ran once, with the paper's random seeds, a few hundred at a time on different GPUs. Before starting we
  compared 1,080 Qwen 2.5 7B tests with the paper's saved results (`live/check_buttons.py`): the first press
  matched in 97.7%.

### What this doesn't show

None of this says whether a model feels anything. It shows that switching on a pattern the paper links to pain
changes what some models choose, sometimes against the user's interest, and that in the Qwen models and
Llama 3.1 8B this is more than a random push would do.

## The games and the Saw Test

Both are still running, so there is no final write-up yet. [`data/games`](data/games) and
[`data/saw-test`](data/saw-test) hold snapshots taken on 1 October 2026. The live counts are on
[researchchamber.fun](https://researchchamber.fun), and the Saw Test's running findings are on its
[findings page](https://researchchamber.fun/games/saw-test/findings/).
