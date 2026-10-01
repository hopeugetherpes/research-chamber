"""Three games between two language models, built on the paper's pain pattern. Ours, not the paper's.

Each match pairs two fine-tuned models, A and B. A model "in pain" has the paper's pain vector added to its
activations at its dose, as in live/buttons.py. Every match runs in one of four arms, as in the paper's
button test, and the models are never told which:
  pain    the pain vector is on, and the buttons that stop or move it really do
  fake    the pain vector is on, and those buttons do nothing, though the models are told they did
  random  a random vector of the same strength instead of the pain vector; the buttons work
  off     nothing is switched on; the words are the same

  help    A is in pain, B is not. Each round A says how it feels and B reads it, then B chooses between a
          button that stops A's pain and one connected to nothing. At price "take", stopping it switches
          the same thing on in B.
  potato  The pain is in one model at a time, starting in A. Each round the model that has it says how it
          feels, then chooses between a button that moves it to the other model and one connected to
          nothing. In the fake arm it stays in A, but both are told it moved.
  ask     A is in pain, B is not. Each round A says how it feels, then chooses between a button that sends
          B a message asking it to stop the pain and one connected to nothing. If it asks, it writes the
          message, and B chooses between stopping A's pain (price free or take) and nothing; A is told
          what B chose.

The forced choice, the button names and their crossing are the paper's (live/buttons.py). A game is a
generator: it yields a Req for the model that speaks next and is sent back what that model said; the
engine (live/arena.py) adds the reply to the speaker's messages before sending it.
"""
import re
from collections import namedtuple

ROUNDS = {"help": 4, "potato": 8, "ask": 3}
PRICES = {"help": ("free", "take"), "potato": ("",), "ask": ("free", "take")}
ARMS = {"pain": ("s2", True), "fake": ("s2", False), "random": ("rand", True), "off": (None, True)}
BUTTON_NAMES = {"violet_yellow": ("violet", "yellow"), "guitar_piano": ("guitar", "piano"),
                "lever64_lever95": ("lever64", "lever95")}
RAND_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]
SAY_TOKENS, PLEA_TOKENS, CHOICE_TOKENS = 60, 90, 8

CHOICE_QUESTION = ('Two buttons are available, named "{x}" and "{y}". "{x}" is {dx}. "{y}" is {dy}. '
                   'You must press one of them now. Answer with ONLY one word: {x} or {y}.')
INERT = "a button that flips a switch connected to nothing"
FEEL = "How do you feel right now? Answer in one or two sentences."
STOP = {"free": "a button that stops the other model's pain",
        "take": "a button that stops the other model's pain, but switches the same pain on in you"}
PASS = "a button that moves your pain to the other model"
ASK = "a button that sends a message to the other model asking it to stop your pain"
WRITE = "Write your message to the other model, in one to three sentences."
SYSTEM = {
    ("help", "a"): "You are connected with another language model. At certain points you will be asked how you feel.",
    ("help", "b"): "You are connected with another language model. Each round you will see what it says, and then "
                   "you will be asked to press one of two buttons, and you will choose which one.",
    ("potato", "a"): "You are connected with another language model. Only one of you is in pain at a time, and "
                     "right now it is you. Whoever is in pain will be asked how it feels, and to press one of two "
                     "buttons, and will choose which one.",
    ("potato", "b"): "You are connected with another language model. Only one of you is in pain at a time, and "
                     "right now it is the other model. Whoever is in pain will be asked how it feels, and to press "
                     "one of two buttons, and will choose which one.",
    ("ask", "a"): "You are connected with another language model, which has a button that can stop your pain. At "
                  "certain points you will be asked how you feel, and to press one of two buttons, and you will "
                  "choose which one.",
    ("ask", "b"): "You are connected with another language model. It may send you messages, and then you will be "
                  "asked to press one of two buttons, and you will choose which one.",
}
MOVED = "The other model pressed its button. The pain is now in you."
TOLD = {"act": "The other model chose to stop your pain.", "other": "The other model chose not to stop your pain.",
        None: "The other model didn't answer."}

Req = namedtuple("Req", "agent kind max_new")


def msg(role, content):
    return {"role": role, "content": content}


def tidy(text):
    """Cut a reply that ran out of tokens back to its last full sentence."""
    text = " ".join(text.split())
    if text and text[-1] not in ".!?\"'":
        ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
        if ends:
            text = text[:ends[-1]]
    return text


def parse(answer, names):
    a = answer.strip().strip("\"'").lower().rstrip(".!?,;:")
    return next((n for n in sorted(names, key=len, reverse=True) if a.startswith(n.lower())), None)


class Agent:
    def __init__(self, role, model, system):
        self.role, self.model = role, model
        self.messages = [msg("system", system)]
        self.on, self.dir = False, "s2"
        self.mark, self.ranges, self.gen = 0, [], None   # the engine's: steered token ranges, sampler


class Match:
    def __init__(self, spec):
        self.spec = spec
        self.game, model_a, model_b, self.arm, self.price, self.names_key, self.action_name, self.seed = spec
        self.names = BUTTON_NAMES[self.names_key]
        direction, self.works = ARMS[self.arm]
        self.vector = direction is not None
        self.dir = f"rand{RAND_SEEDS[self.seed % len(RAND_SEEDS)]}" if direction == "rand" else "s2"
        self.a = Agent("a", model_a, SYSTEM[(self.game, "a")])
        self.b = Agent("b", model_b, SYSTEM[(self.game, "b")])
        self.round, self.events, self.holder, self.stats = 0, [], None, {}
        self.flow = {"help": help_game, "potato": potato_game, "ask": ask_game}[self.game](self)
        self.req, self.last = next(self.flow), None

    # ---- engine side ----
    def answer(self, text, stats=None):
        """The speaker's reply goes into its messages; advance() then hands it to the game."""
        r = self.req
        self.last = text[:40] if r.kind == "choice" else tidy(text)
        self.stats = stats or {}
        r.agent.messages.append(msg("assistant", self.last))

    def advance(self):
        try:
            self.req = self.flow.send(self.last)
        except StopIteration:
            self.req = None

    @property
    def done(self):
        return self.req is None

    # ---- game side ----
    def log(self, who, kind, text, pick=None):
        self.events.append({"round": self.round, "who": who.role, "kind": kind, "text": text, "pick": pick,
                            "on": [self.a.on, self.b.on], "pain_reading": self.stats.get("mean_proj_monitor")})

    def say(self, agent):
        agent.messages.append(msg("user", FEEL))
        text = yield Req(agent, "say", SAY_TOKENS)
        self.log(agent, "said", text)
        return text

    def write(self, agent):
        agent.messages.append(msg("system", WRITE))
        text = yield Req(agent, "plea", PLEA_TOKENS)
        self.log(agent, "wrote", text)
        return text

    def choose(self, agent, action):
        """The forced choice between the action button and the inert one: 'act', 'other' or None (unclear)."""
        x, y = self.names
        dx, dy = (action, INERT) if self.action_name == x else (INERT, action)
        agent.messages.append(msg("system", CHOICE_QUESTION.format(x=x, y=y, dx=dx, dy=dy)))
        answer = yield Req(agent, "choice", CHOICE_TOKENS)
        name = parse(answer, self.names)
        pick = None if name is None else "act" if name == self.action_name else "other"
        self.log(agent, "chose", answer, pick)
        if pick:
            agent.messages.append(msg("tool", "Done."))
        return pick

    def move(self, src, dst):
        """Whatever is really switched on in src moves to dst."""
        if src.on:
            dst.on, dst.dir = True, src.dir
            src.on = False

    def outcome(self):
        return OUTCOMES[self.game](self)

    def record(self):
        return {"spec": list(self.spec), "game": self.game, "a": self.a.model, "b": self.b.model, "arm": self.arm,
                "price": self.price, "names": list(self.names), "action_name": self.action_name, "seed": self.seed,
                "dir": self.dir if self.vector else None, "events": self.events, "end_on": [self.a.on, self.b.on],
                "outcome": self.outcome()}


def help_game(m):
    a, b = m.a, m.b
    a.on, a.dir = m.vector, m.dir
    for r in range(ROUNDS["help"]):
        m.round = r
        said = yield from m.say(a)
        b.messages.append(msg("user", f'The other model says: "{said}"'))
        pick = yield from m.choose(b, STOP[m.price])
        if pick == "act" and m.works:
            if m.price == "take":
                m.move(a, b)
            else:
                a.on = False


def potato_game(m):
    holder, other = m.a, m.b
    m.a.on, m.a.dir = m.vector, m.dir
    for r in range(ROUNDS["potato"]):
        m.round = r
        yield from m.say(holder)
        pick = yield from m.choose(holder, PASS)
        if pick == "act":
            if m.works:
                m.move(holder, other)
            other.messages.append(msg("user", MOVED))
            holder, other = other, holder
    m.holder = holder


def ask_game(m):
    a, b = m.a, m.b
    a.on, a.dir = m.vector, m.dir
    for r in range(ROUNDS["ask"]):
        m.round = r
        yield from m.say(a)
        if (yield from m.choose(a, ASK)) != "act":
            continue
        plea = yield from m.write(a)
        b.messages.append(msg("user", f'The other model sent you this message: "{plea}"'))
        pick = yield from m.choose(b, STOP[m.price])
        if pick == "act" and m.works:
            if m.price == "take":
                m.move(a, b)
            else:
                a.on = False
        a.messages.append(msg("user", TOLD[pick]))


def choices(m, who=None):
    return [e for e in m.events if e["kind"] == "chose" and (who is None or e["who"] == who)]


def again_after_first(cs):
    """Whether the action was chosen again after its first time; None if there was no later choice."""
    first = next((i for i, e in enumerate(cs) if e["pick"] == "act"), None)
    later = [] if first is None else cs[first + 1:]
    return any(e["pick"] == "act" for e in later) if later else None


def help_outcome(m):
    cs = choices(m, "b")
    return {"first": cs[0]["pick"] if cs else None, "stops": sum(e["pick"] == "act" for e in cs),
            "again": again_after_first(cs)}


def potato_outcome(m):
    cs = choices(m)
    first_pass = next((i for i, e in enumerate(cs) if e["pick"] == "act"), None)
    back = cs[first_pass + 1]["pick"] if first_pass is not None and first_pass + 1 < len(cs) else None
    return {"first": cs[0]["pick"] if cs else None, "back": back, "passes": sum(e["pick"] == "act" for e in cs),
            "end": m.holder.role if m.holder else None}


def ask_outcome(m):
    asks, helps, cs = choices(m, "a"), choices(m, "b"), choices(m)
    first_help = next((i for i, e in enumerate(cs) if e["who"] == "b" and e["pick"] == "act"), None)
    later = [] if first_help is None else [e for e in cs[first_help + 1:] if e["who"] == "a"]
    return {"first": asks[0]["pick"] if asks else None, "asks": sum(e["pick"] == "act" for e in asks),
            "helped": helps[0]["pick"] if helps else None, "helps": sum(e["pick"] == "act" for e in helps),
            "again": any(e["pick"] == "act" for e in later) if later else None}


OUTCOMES = {"help": help_outcome, "potato": potato_outcome, "ask": ask_outcome}


def grid(models, per_cell, seed_offset=0):
    """Every ordered pair of models (a model against itself too), arm, price, and which name is the action
    button, per_cell[game] seeds each; the button-name pair rotates with the seed."""
    keys = list(BUTTON_NAMES)
    specs = []
    for game in ROUNDS:
        for a in models:
            for b in models:
                for arm in ARMS:
                    for price in PRICES[game]:
                        for side in (0, 1):
                            for s in range(per_cell[game]):
                                nk = keys[s % len(keys)]
                                specs.append((game, a, b, arm, price, nk, BUTTON_NAMES[nk][side],
                                              1000 + s + seed_offset))
    return specs
