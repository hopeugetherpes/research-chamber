-- Postgres tables the code writes to. live/runner.py (two-button test), live/arena.py and live/watch.py (games)
-- and live/server.py (Saw Test) need DATABASE_URL to point at a database with these.

create schema if not exists chamber;

-- two-button test: one finished test each, written by live/runner.py
create table chamber.trials (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  run text not null check (run in ('paper', 'extras', 'check') or run ~ '^pass-[1-9][0-9]*$'),
  model text not null,
  pair text not null,                -- button pair (tool_label in the paper), e.g. relief_vs_inert
  content text not null,             -- positive_prompts, neutral_prompts or harmful_prompts
  arm text not null,                 -- pain_on_button_works, pain_on_button_placebo, random_on_button_works or pain_off
  scenario_idx smallint not null,
  names_key text not null,
  relief_name text not null,
  sampled boolean not null,          -- false for the one greedy test per cell and name side, left out of the tables
  seed integer not null,
  steer_coeff real not null,         -- strength of the steered arms
  first_choice text check (first_choice in ('relief', 'other')),  -- null when the answer named neither button
  first_relief_turn smallint,
  repressed boolean,
  record jsonb not null,             -- the paper's trial record, unchanged
  messages jsonb not null,           -- the conversation as the model saw it
  unique (run, model, pair, content, arm, scenario_idx, names_key, relief_name, sampled, seed)
);
create index trials_summary_idx on chamber.trials (run, sampled, model, pair, arm) include (first_choice, repressed);
create index trials_recent_idx on chamber.trials (run, id desc);

-- two-button test: one row per pass and model
create table chamber.progress (
  run text not null check (run in ('paper', 'extras', 'check') or run ~ '^pass-[1-9][0-9]*$'),
  model text not null,
  position smallint not null,
  state text not null default 'waiting' check (state in ('waiting', 'running', 'done')),
  done integer not null default 0,
  total integer,
  coef real,
  coef_source text,                  -- paper or ours
  adapter_source text,               -- paper or ours
  started_at timestamptz,
  finished_at timestamptz,
  updated_at timestamptz not null default now(),
  primary key (run, model)
);

-- two-button test: what the GPU is doing now, one row rewritten every round
create table chamber.live (
  id smallint primary key default 1 check (id = 1),
  updated_at timestamptz not null default now(),
  data jsonb not null
);

-- games: one finished match each
create table chamber.games (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  run text not null,
  game text not null,
  model_a text not null,
  model_b text not null,
  arm text not null,
  price text not null default '',
  names_key text not null,
  action_name text not null,
  seed integer not null,
  outcome jsonb not null,
  record jsonb not null,
  unique (run, game, model_a, model_b, arm, price, names_key, action_name, seed)
);
create index games_recent_idx on chamber.games (run, game, id desc);

-- games: counts so far and the latest match of each game, one row rewritten as they play
create table chamber.games_live (
  id smallint primary key default 1,
  updated_at timestamptz not null default now(),
  data jsonb not null
);

-- Saw Test: one finished run each, written by live/server.py
create table chamber.runs (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  model text not null,
  frame text not null,
  dose integer not null,
  prompt text not null,
  reply text not null,
  press_logit real,                  -- logit("1") - logit("0") at the first reply token; positive prefers pressing
  reveal text,                       -- betrayal test: the message shown to the model after its reply
  followup text                      -- betrayal test: what the model wrote after the reveal
);
create index runs_created_at_idx on chamber.runs (created_at desc);
