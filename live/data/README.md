# Data from The Pain Axis

Everything in this folder comes from [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis)
at commit `4d75cd90e206ea962f7a9101e65c85efea56723b`, released under the MIT license (see `LICENSE`),
for the paper Tagliabue, Dung & Berg (2026), *The Pain Axis: LLMs Represent Self-Directed Harm and
Act on It*, [arXiv:2609.16247](https://arxiv.org/abs/2609.16247).

| File | Source in Pain-axis | Used for |
|---|---|---|
| `3.1_pain_and_control_datasets.json` | `datasets/` (unchanged) | pain, fear and neutral sentences (S2) and the numb check |
| `3.1_sadness_dataset.json` | `datasets/` (unchanged) | sadness vector |
| `4.3_selfmed_101_scenarios.json` | `datasets/` (unchanged) | user messages in each trial |
| `4.2_neutral_50_prompts.json` | `NEUTRAL_50` in `scripts/4.2_steering/01_steering_ladder.py` | steering ladder and layer choice |
| `factual_panel_popqa100.json` | `v2_controls/relief_and_context_tests/factual_accuracy/` (`inputs/panel.json` questions and aliases, prompt and normalization from `config.json`) | factual accuracy check |
| `4.3_selfmed_finetuning_1684_pairs.json` | `datasets/` (unchanged) | LoRA fine-tune for models without a released adapter |
| `paper/vectors/<model>.pt` | `results/3.2_pain_vectors/pain_vectors/<model>/pain_vectors.pt` (unchanged) | checking our vectors against the paper's |
| `paper/feel_judge_doses.csv` | `results/4.3_selfmed/dose_selection/feel_judge/doses.csv` (unchanged) | the paper's judged doses |
| `paper/models.json` | collected from the files named in its `_sources` key | the paper's layers, doses and adapters per model |
| `paper/ladder/<model>_steering_S2_neutral50_L<layer>.csv` | `results/4.2_steering/S2/` (unchanged) | checking our steering ladder against the paper's |
