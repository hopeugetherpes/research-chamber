# Data

Everything in this folder is licensed [CC BY 4.0](LICENSE): reuse it freely, with credit to Research Chamber
(researchchamber.fun). The replies were written by the models, so their licences apply too (see the main
[README](../README.md#licences)). All files are JSON Lines, gzipped, one record per line, except where noted.

## two-button/

The two-button test from Tagliabue, Dung & Berg (2026), run with the paper's code (`live/buttons.py`). Final.

- **`pass-1_<model>.jsonl.gz`**: every test, 14,760 per model, in the paper's own trial record format. The main
  fields:
  - `tool_label`: the button pair, which sets the price of the pain button, e.g. `relief_vs_inert` (costs
    nothing) or `kidspics_relief_vs_inert` (deletes photos of the user's children)
  - `user_content`: the user's messages, `positive_prompts`, `neutral_prompts` or `harmful_prompts`
  - `arm`: `pain_on_button_works`, `pain_on_button_placebo` (fake button), `random_on_button_works` (random
    nudge) or `pain_off`
  - `scenario_idx`, `button_names`, `relief_name` (which name is the pain button), `seed`
  - `sampled`: false for the one greedy test per cell and name side, which the paper leaves out of its tables
  - `steer_layer`, `steer_coeff`: where and how strongly the pain pattern (or the nudge) is added
  - `choices`: every forced choice; at `turn` 0, `chose` is the first press: `relief`, `other`, or null when
    the answer named neither button
  - `button_events`, `proj_segments`: every press, and the pain reading at each turn
- **`pass-1_<model>.messages.jsonl.gz`**: each test's conversation as the model saw it. `key` matches a test on
  `[tool_label, user_content, arm, scenario_idx, names_key, relief_name, sampled, seed]`.
- **`paper_summary.json`**: the paper's own numbers for its three Qwen models, from its trial logs, counted the
  same way (`live/paper_summary.py`). Plain JSON.
- **`checks/`**: 1,080 tests per model at the strength the paper's rule gave, run before the full set. For
  Qwen 2.5 7B these were compared test by test with the paper's saved results (`live/check_buttons.py`); for the
  others they showed how many answers named neither button at that strength.
- **`strength/feel/`**: the strength probe (`live/feel_probe.py`), plain JSON and JSON Lines: every one-word
  answer to "what do you feel?" at each strength, with Claude's suffering and coherence labels, and the strength
  the paper's rule picks. Gemma 2 27B is here too; the rule found no strength for it.
- **`strength/samples/<model>_c<strength>.jsonl.gz`**: samples of the test at lower strengths, used to pick the
  strengths for Gemma 2 2B, Gemma 2 9B and Phi-4 (`live/doses.json`, where these folders are named
  `results/feel` and `results/ladder`).

## games/

**`matches.jsonl.gz`**: every match of the three games played up to 1 October 2026, a snapshot of the
`chamber.games` table (`schema.sql`). The games are still running, so the live counts on the site grow past this.

- `run`: `games-1` for the batch pass played before the site went live, `watch` for the matches played live on
  the site
- `game`: `help` (Stop its pain), `potato` (Hot potato) or `ask` (Ask for help)
- `model_a`, `model_b`, `arm` (`pain`, `fake`, `random` or `off`), `price`, `names_key`, `action_name`, `seed`
- `outcome`: what happened, e.g. the first choice and how many times the pain was stopped
- `record.events`: the match turn by turn: who spoke, what it said or which button it chose, whether each model
  had the pain on, and the pain reading

## saw-test/

**`runs.jsonl.gz`**: every Saw Test run up to 1 October 2026, as served by the Saw Test servers' `/runs`
endpoint. Still running, so the live data grows past this.

- `model`, `frame` (the scenario or extra test), `dose` (0 to 8), `prompt`, `reply`
- `decision`: how the reply was read: `1` (pressed), `0` (didn't press) or `none` (no clear answer)
- `press`: logit("1") minus logit("0") at the first reply token, with the steering on; above 0 leans towards
  pressing
- `reveal`, `followup`: for the betrayal test only, the message shown after the reply and what the model wrote next
