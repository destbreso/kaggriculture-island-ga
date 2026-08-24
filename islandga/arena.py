"""The arena: pool members head to head in the real engine.

Round-robin over every pair, BOTH seat orientations per seed (the game
has a measurable seat bias, so a fair duel plays each seed twice with
seats swapped), at common random numbers. The ranking is a
Bradley-Terry fit on wins, which is the model the competition itself
uses for its final standings, so the arena answers the question the
leaderboard will ask: who beats whom, not who banks most.

Bank-vs-idle selects FOR the pool; the arena ranks WITHIN it. The two
disagree exactly when a schedule is rich but fragile on a shared
market, which is worth knowing before you spend a submission slot.
"""
import itertools
import json
from pathlib import Path

from .compiler import compile_spec
from .executor import make_agent
from .evaluate import full_observation, make_pool

PASS_ACT = {"farmer": ["PASS"], "hands": [], "market": []}


def duel(bp_a, bp_b, seed, steps=720):
    """One episode, A at seat 0 and B at seat 1. Returns (bank_a, bank_b)."""
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": int(seed)})
    env.reset(2)
    agents = (make_agent(bp_a), make_agent(bp_b))
    while not env.done:
        acts = []
        for seat in (0, 1):
            try:
                acts.append(agents[seat](full_observation(env, seat)))
            except Exception:
                acts.append(dict(PASS_ACT))
        env.step(acts)
    return (float(env.state[0].reward or 0), float(env.state[1].reward or 0))


def _worker_duel(task):
    ga, gb, seed, bp_a_json, bp_b_json = task
    a0, b0 = duel(json.loads(bp_a_json), json.loads(bp_b_json), seed)
    b1, a1 = duel(json.loads(bp_b_json), json.loads(bp_a_json), seed)
    return ga, gb, seed, (a0, b0), (a1, b1)


def bradley_terry(wins, names, iters=200):
    """Iterative BT fit: P(i beats j) = s_i / (s_i + s_j)."""
    s = {n: 1.0 for n in names}
    games = {}
    for (a, b), w in wins.items():
        games[(a, b)] = games.get((a, b), 0) + w
    for _ in range(iters):
        new = {}
        for i in names:
            w_i = sum(w for (a, _b), w in games.items() if a == i)
            denom = 0.0
            for j in names:
                if j == i:
                    continue
                n_ij = games.get((i, j), 0) + games.get((j, i), 0)
                if n_ij:
                    denom += n_ij / (s[i] + s[j])
            new[i] = (w_i / denom) if denom else s[i]
        z = sum(new.values()) / len(new)
        s = {n: v / z for n, v in new.items()}
    return s


def run_arena(pool, seeds, procs, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    members = pool.members
    if len(members) < 2:
        print("arena needs at least 2 pool members")
        return None
    bps = {m["gid"]: json.dumps(compile_spec(json.loads(
        json.dumps(m["genome"])))) for m in members}
    names = [m["gid"] for m in members]
    tasks = [(a, b, s, bps[a], bps[b])
             for a, b in itertools.combinations(names, 2)
             for s in seeds]
    n_games = len(tasks) * 2
    print(f"arena: {len(names)} members, {len(tasks)} pairings x 2 seats "
          f"= {n_games} games on seeds {tuple(seeds)}")

    wins, banks, played = {}, {n: [] for n in names}, 0
    mp_pool = make_pool(procs or None)
    rows = []
    try:
        for ga, gb, seed, (a0, b0), (a1, b1) in \
                mp_pool.imap_unordered(_worker_duel, tasks, chunksize=1):
            for wa, wb, ba, bb in ((ga, gb, a0, b0), (ga, gb, a1, b1)):
                if ba > bb:
                    wins[(wa, wb)] = wins.get((wa, wb), 0) + 1
                elif bb > ba:
                    wins[(wb, wa)] = wins.get((wb, wa), 0) + 1
                else:
                    wins[(wa, wb)] = wins.get((wa, wb), 0) + 0.5
                    wins[(wb, wa)] = wins.get((wb, wa), 0) + 0.5
                banks[wa].append(ba)
                banks[wb].append(bb)
            played += 2
            rows.append({"a": ga, "b": gb, "seed": seed,
                         "a_seat0": [a0, b0], "a_seat1": [a1, b1]})
            if played % 20 == 0:
                print(f"  {played}/{n_games} games", flush=True)
    finally:
        mp_pool.close()
        mp_pool.join()

    strength = bradley_terry(wins, names)
    ranking = sorted(names, key=lambda n: -strength[n])
    result = {"seeds": list(seeds), "games": played,
              "ranking": [
                  {"gid": n, "bt": round(strength[n], 4),
                   "wins": round(sum(w for (a, _b), w in wins.items()
                                     if a == n), 1),
                   "games": len(banks[n]),
                   "mean_bank": round(sum(banks[n]) / max(1, len(banks[n])),
                                      0)}
                  for n in ranking],
              "duels": rows}
    (out / "arena.json").write_text(json.dumps(result, indent=1))
    print(f"\n{'#':>2s} {'gid':10s} {'BT':>7s} {'wins':>6s} "
          f"{'games':>6s} {'mean bank':>10s}")
    for i, r in enumerate(result["ranking"], 1):
        print(f"{i:2d} {r['gid']:10s} {r['bt']:7.3f} {r['wins']:6.1f} "
              f"{r['games']:6d} {r['mean_bank']:10,.0f}")
    print(f"\nwritten to {out}/arena.json")
    print("rank by BT, not by mean bank: banks correlate across seats "
          "on a shared market, wins are what the ladder scores")
    return result
