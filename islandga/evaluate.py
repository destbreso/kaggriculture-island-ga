"""Evaluation: bank vs an idle opponent in the real engine, at COMMON
RANDOM NUMBERS.

The engine seed reproduces weeds and shop draws, so evaluating every
candidate on the same fixed seed panel makes every comparison paired:
differences are the candidate's, not the world's. That is what lets a
small panel rank genomes at all.

Requires ``kaggle-environments`` (the competition engine ships in it).
"""
import json
import multiprocessing as mp

from .compiler import compile_spec
from .executor import make_agent

PASS_ACT = {"farmer": ["PASS"], "hands": [], "market": []}


def full_observation(env, seat):
    """Seats above 0 get a trimmed observation; merge the shared fields
    the way the live framework does."""
    shared = env.state[0].observation
    obs = dict(env.state[seat].observation)
    for key, value in shared.items():
        if obs.get(key) is None and key != "player":
            obs[key] = value
    return obs


def bank_of(bp, seed, steps=720):
    """One episode: the blueprint's agent (seat 0) vs an idle opponent."""
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": int(seed)})
    env.reset(2)
    agent = make_agent(bp)
    while not env.done:
        try:
            ours = agent(full_observation(env, 0))
        except Exception:
            ours = dict(PASS_ACT)
        env.step([ours, dict(PASS_ACT)])
    return float(env.state[0].reward or 0)


def eval_genome(genome, seeds):
    """Sequential evaluation of one genome on a seed panel."""
    bp = compile_spec(json.loads(json.dumps(genome)))
    return {s: bank_of(bp, s) for s in seeds}


# ---------------------------------------------------------------- pool
def _worker_eval(task):
    gid, bp_json, seed = task
    bp = json.loads(bp_json)
    return gid, seed, bank_of(bp, seed)


def make_pool(procs=None):
    procs = procs or max(1, (mp.cpu_count() or 4) - 2)
    return mp.get_context("spawn").Pool(procs)


def eval_many(pool, tasks):
    """tasks: [(gid, bp_json, seed)] -> yields (gid, seed, bank)."""
    return pool.imap_unordered(_worker_eval, tasks, chunksize=1)
