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
from .pool import Pool


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
    pool_file: str = "results/pool.json"
    pool_size: int = 10
    islands: int = 0                   # 0 = one per species; more cycles them
    stagnation_gens: int = 10          # island restart after N flat gens
    global_stagnation_gens: int = 18   # archipelago restart after M flat gens
    immigrant_every: int = 4           # fresh random genome cadence, 0 = off

    @classmethod
    def load(cls, path):
        cfg = cls(**json.loads(Path(path).read_text()))
        return cfg


def run_search(cfg: SearchConfig, out_dir, compiler=None,
               pool_factory=None, eval_stream=None):
    """The orchestration is executor-agnostic: inject `compiler`
    (genome -> blueprint), `pool_factory` (procs -> worker pool) and
    `eval_stream` (pool, tasks -> (gid, seed, bank) iterator) to run
    the same search over a different executor. Defaults use the
    bundled reference executor."""
    compiler = compiler or compile_spec
    pool_factory = pool_factory or make_pool
    eval_stream = eval_stream or eval_many
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.rng_seed)
    names = list(cfg.species)
    n_isl = cfg.islands or len(names)
    isl_species = [names[i % len(names)] for i in range(n_isl)]
    islands = []
    for name in isl_species:
        base = make_species(name, rng)
        islands.append([base] + [mutate(base, rng)
                                 for _ in range(cfg.pop - 1)])
    isl_stall = [0] * n_isl            # gens since the island last improved
    isl_prev = [float("-inf")] * n_isl
    global_stall, prev_global, restarts = 0, float("-inf"), 0

    cache = {}                         # gid -> {seed: bank}
    genomes = {}
    top_pool = Pool(cfg.pool_file, cfg.pool_size) if cfg.pool_file else None
    log_f = open(out / "log.jsonl", "a")
    pool = pool_factory(cfg.procs or None)
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
                        bp_json = json.dumps(compiler(copy.deepcopy(g)))
                        for s in cfg.seeds:
                            if s not in cache.get(gid, {}):
                                tasks.append((gid, bp_json, s))
            t0 = time.time()
            gid_isl = {gid_of(g): ii for ii, pop in enumerate(islands)
                       for g in pop}
            for gid, seed, bank in eval_stream(pool, tasks):
                cache.setdefault(gid, {})[seed] = bank
                games += 1
                log_f.write(json.dumps(
                    {"gen": gen, "island": gid_isl.get(gid), "gid": gid,
                     "seed": seed, "bank": bank}) + "\n")
            log_f.flush()

            # rotate the top-N snapshot pool with this generation's crop
            if top_pool is not None:
                for pop in islands:
                    for g in pop:
                        gid = gid_of(g)
                        if fitness(gid) is not None:
                            top_pool.offer(g, cache[gid],
                                           origin=f"{out.name}:g{gen}")

            new_islands = []
            for ii, pop in enumerate(islands):
                scored = sorted(pop,
                                key=lambda g: -(fitness(gid_of(g)) or 0))
                fbest = fitness(gid_of(scored[0])) or 0
                if fbest > best_ever[1]:
                    best_ever = (scored[0], fbest)
                # stagnation bookkeeping: improvement means beating the
                # island's own best, not merely holding it (the elite
                # always holds it)
                if fbest > isl_prev[ii] + 1:
                    isl_stall[ii] = 0
                else:
                    isl_stall[ii] += 1
                isl_prev[ii] = max(isl_prev[ii], fbest)
                if isl_stall[ii] >= cfg.stagnation_gens:
                    # CATACLYSM: keep the elite, reseed the island around
                    # it with fresh species draws and hot mutants of the
                    # elite. A stuck island repeats this every N flat gens,
                    # which is the intended behaviour: keep kicking.
                    elite = scored[0]
                    n_fresh = max(1, len(pop) // 3)
                    fresh = [make_species(isl_species[ii], rng)
                             for _ in range(n_fresh)]
                    hot = [mutate(elite, rng, moves=(3, 6))
                           for _ in range(len(pop) - 1 - n_fresh)]
                    new_islands.append([elite] + fresh + hot)
                    isl_stall[ii] = 0
                    restarts += 1
                    print(f"  island {ii} ({isl_species[ii]}) RESTARTED: "
                          f"flat {cfg.stagnation_gens} gens at "
                          f"{fbest:,.0f}", flush=True)
                    continue
                # hypermutation: a half-stagnant island mutates hotter
                hot_moves = (2, 5) if isl_stall[ii] >= \
                    max(2, cfg.stagnation_gens // 2) else (1, 3)
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
                    nxt.append(mutate(child, rng, moves=hot_moves))
                # random immigrants: constant cheap diversity
                if cfg.immigrant_every and gen % cfg.immigrant_every == 0 \
                        and len(nxt) > 2:
                    nxt[-1] = make_species("random", rng)
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

            # archipelago-level escape: when the GLOBAL best has been flat
            # for long enough, only the island holding it survives and the
            # rest are reborn from their species
            if best_ever[1] > prev_global + 1:
                global_stall = 0
            else:
                global_stall += 1
            prev_global = max(prev_global, best_ever[1])
            if global_stall >= cfg.global_stagnation_gens:
                keep = max(range(len(islands)), key=lambda i: max(
                    (fitness(gid_of(g)) or 0) for g in islands[i]))
                for i in range(len(islands)):
                    if i == keep:
                        continue
                    base = make_species(isl_species[i], rng)
                    islands[i] = [base] + [mutate(base, rng)
                                           for _ in range(cfg.pop - 1)]
                    isl_stall[i] = 0
                    isl_prev[i] = float("-inf")
                global_stall = 0
                restarts += 1
                print(f"  ARCHIPELAGO RESTART: global best flat "
                      f"{cfg.global_stagnation_gens} gens; island {keep} "
                      f"kept, the rest reseeded", flush=True)

            fits = [round(max((fitness(gid_of(g)) or 0) for g in p))
                    for p in islands]
            print(f"gen {gen:3d}  islands {fits}  "
                  f"best {best_ever[1]:,.0f}  games {games:,}  "
                  f"stall {global_stall}  "
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
            bp_json = json.dumps(compiler(copy.deepcopy(g)))
            conf = {}
            for gid, seed, bank in eval_stream(
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
                "pool_file": cfg.pool_file or None,
                "pool_members": len(top_pool.members) if top_pool else 0,
                "restarts": restarts,
            }
            (out / "best.json").write_text(json.dumps(metrics, indent=1))
            print("\n=== FINAL METRICS " + "=" * 44)
            for k in ("best_gid", "screen_mean", "confirm_mean",
                      "winners_curse_gap", "gen1_best",
                      "improvement_vs_gen1", "generations", "games",
                      "wall_hours", "island_final_bests", "restarts"):
                print(f"  {k:22s} {metrics[k]}")
            print(f"  written to {out}/best.json  (genome included)")
            print("  quote the CONFIRM mean, never the screen mean",
                  flush=True)
        pool.close()
        pool.join()
        log_f.close()
    return metrics
