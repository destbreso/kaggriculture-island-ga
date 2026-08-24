"""The island GA: species, tournament selection, block crossover, ring
migration, and the screen/confirm discipline.

Everything a run produces lands in its output directory:

* ``log.jsonl``    one row per engine game (gen, island, gid, seed, bank)
* ``state.json``   populations and the best-so-far, written every gen
* ``best.json``    final metrics: the winning genome, its screen banks,
                   and its CONFIRM banks on a disjoint seed panel

The confirm panel is not optional decoration. The screening mean of a
selected best is inflated by construction (the winner's curse); the
number to quote is the disjoint-panel mean, and the run prints both so
the gap itself is visible.
"""
import copy
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .compiler import compile_spec
from .evaluate import eval_many, make_pool
from .genome import crossover, gid_of, make_species, mutate


@dataclass
class SearchConfig:
    species: tuple = ("envelope", "boundmix", "intensity", "random")
    pop: int = 8
    seeds: tuple = (11, 23, 47)
    confirm_seeds: tuple = (5, 13, 29, 61, 83)
    migrate_every: int = 6
    tournament: int = 3
    crossover_p: float = 0.6
    hours: float = 1.0
    procs: int = 0                     # 0 = cpu count - 2
    rng_seed: int = 20260824

    @classmethod
    def load(cls, path):
        cfg = cls(**json.loads(Path(path).read_text()))
        return cfg


def run_search(cfg: SearchConfig, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.rng_seed)
    islands = []
    for name in cfg.species:
        base = make_species(name, rng)
        islands.append([base] + [mutate(base, rng)
                                 for _ in range(cfg.pop - 1)])

    cache = {}                         # gid -> {seed: bank}
    genomes = {}
    log_f = open(out / "log.jsonl", "a")
    pool = make_pool(cfg.procs or None)
    t_start = time.time()
    t_end = t_start + cfg.hours * 3600
    gen, games = 0, 0
    best_ever = (None, float("-inf"))
    gen1_best = None

    def fitness(gid):
        r = cache.get(gid, {})
        if all(s in r for s in cfg.seeds):
            return sum(r[s] for s in cfg.seeds) / len(cfg.seeds)
        return None

    try:
        while time.time() < t_end:
            gen += 1
            tasks = []
            for ii, pop in enumerate(islands):
                for g in pop:
                    gid = gid_of(g)
                    genomes[gid] = g
                    if fitness(gid) is None:
                        bp_json = json.dumps(compile_spec(copy.deepcopy(g)))
                        for s in cfg.seeds:
                            if s not in cache.get(gid, {}):
                                tasks.append((gid, bp_json, s))
            t0 = time.time()
            gid_isl = {gid_of(g): ii for ii, pop in enumerate(islands)
                       for g in pop}
            for gid, seed, bank in eval_many(pool, tasks):
                cache.setdefault(gid, {})[seed] = bank
                games += 1
                log_f.write(json.dumps(
                    {"gen": gen, "island": gid_isl.get(gid), "gid": gid,
                     "seed": seed, "bank": bank}) + "\n")
            log_f.flush()

            new_islands = []
            for pop in islands:
                scored = sorted(pop,
                                key=lambda g: -(fitness(gid_of(g)) or 0))
                fbest = fitness(gid_of(scored[0])) or 0
                if fbest > best_ever[1]:
                    best_ever = (scored[0], fbest)
                nxt = [scored[0]]      # elitism
                while len(nxt) < len(pop):
                    def pick():
                        c = rng.sample(scored, min(cfg.tournament,
                                                   len(scored)))
                        return max(c, key=lambda g: fitness(gid_of(g)) or 0)
                    a, b = pick(), pick()
                    child = crossover(a, b, rng) \
                        if rng.random() < cfg.crossover_p \
                        else copy.deepcopy(a)
                    nxt.append(mutate(child, rng))
                new_islands.append(nxt)
            islands = new_islands
            if gen == 1:
                gen1_best = best_ever[1]
            if gen % cfg.migrate_every == 0:
                bests = [max(p, key=lambda g: fitness(gid_of(g)) or 0)
                         for p in islands]
                for i, pop in enumerate(islands):
                    donor = copy.deepcopy(bests[(i - 1) % len(islands)])
                    pop.sort(key=lambda g: -(fitness(gid_of(g)) or 0))
                    pop[-1] = donor

            fits = [round(max((fitness(gid_of(g)) or 0) for g in p))
                    for p in islands]
            print(f"gen {gen:3d}  islands {fits}  "
                  f"best {best_ever[1]:,.0f}  games {games:,}  "
                  f"{time.time()-t0:.0f}s/gen", flush=True)
            (out / "state.json").write_text(json.dumps(
                {"gen": gen, "games": games,
                 "best_gid": gid_of(best_ever[0]) if best_ever[0] else None,
                 "best_fit": best_ever[1], "islands": islands,
                 "config": asdict(cfg)}, default=str))
    finally:
        metrics = None
        if best_ever[0] is not None:
            g = best_ever[0]
            bp_json = json.dumps(compile_spec(copy.deepcopy(g)))
            conf = {}
            for gid, seed, bank in eval_many(
                    pool, [(gid_of(g), bp_json, s)
                           for s in cfg.confirm_seeds]):
                conf[seed] = bank
            conf_mean = sum(conf.values()) / max(1, len(conf))
            metrics = {
                "best_gid": gid_of(g), "genome": g,
                "screen_mean": round(best_ever[1], 1),
                "screen_banks": cache.get(gid_of(g), {}),
                "confirm_mean": round(conf_mean, 1),
                "confirm_banks": conf,
                "winners_curse_gap": round(best_ever[1] - conf_mean, 1),
                "gen1_best": round(gen1_best or 0, 1),
                "improvement_vs_gen1": round(best_ever[1]
                                             - (gen1_best or 0), 1),
                "generations": gen, "games": games,
                "wall_hours": round((time.time() - t_start) / 3600, 3),
                "island_final_bests": [
                    round(max((fitness(gid_of(g2)) or 0) for g2 in p))
                    for p in islands],
            }
            (out / "best.json").write_text(json.dumps(metrics, indent=1))
            print("\n=== FINAL METRICS " + "=" * 44)
            for k in ("best_gid", "screen_mean", "confirm_mean",
                      "winners_curse_gap", "gen1_best",
                      "improvement_vs_gen1", "generations", "games",
                      "wall_hours", "island_final_bests"):
                print(f"  {k:22s} {metrics[k]}")
            print(f"  written to {out}/best.json  (genome included)")
            print("  quote the CONFIRM mean, never the screen mean",
                  flush=True)
        pool.close()
        pool.join()
        log_f.close()
    return metrics
