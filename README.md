# Kaggriculture island GA

Search for your OWN base schedule in Kaggle's
[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
competition, instead of copying one: an island genetic algorithm over a
declarative season spec, a compiler that derives the whole market
channel (financing included), a reference executor, and evaluation in
the real engine at common random numbers, ending in screen/confirm
metrics you can quote.

**The reasoning behind every design choice lives in the companion
notebook,
[Island GA | An owned schedule is a moat](https://www.kaggle.com/code/destbreso/island-ga-an-owned-schedule-is-a-moat):**
why the winning shape on this ladder is a base schedule plus a thin
layer, why the genome must be a spec and never an action stream, why
islands, why the initial species are grown from priors rather than
seeded with known schedules (a takeover simulation included), and the
relaxed bound that prices the whole corridor before you spend a minute
of compute. The notebook keeps the pseudocode; this repo keeps the
versioned code you can actually run with your own parameters.

## The pipeline

```
genome (a 12-number spec)          islandga/genome.py
   │  4 species, 7 macro-mutations, block crossover
   ▼
compiler                           islandga/compiler.py
   │  spec -> per-day targets + the full market channel
   │  (financing derived from scratch: sells, seeds, feed,
   │   herd waves, land, the Fibonacci hire curve)
   ▼
reference executor                 islandga/executor.py
   │  blueprint -> per-turn actions (greedy dispatch + 2 valves)
   ▼
real engine, vs idle, fixed seeds  islandga/evaluate.py
   │  common random numbers: every comparison is paired
   ▼
island GA                          islandga/search.py
      tournament + elitism + ring migration
      -> log.jsonl, state.json, best.json (final metrics,
         with a DISJOINT confirm panel: quote that number)
```

## Quickstart

```bash
pip install -r requirements.txt      # kaggle-environments (the engine)

# 1. price the corridor before searching (runs in seconds)
python cli.py bound

# 2. compile one species and play one real game vs an idle opponent
python cli.py smoke --species envelope --seed 11

# 3. search (1 hour, all cores minus two; scale --hours up when happy)
python cli.py search --hours 1 --out results/run1

# 4. the final metrics, plus a trajectory chart if matplotlib is around
python cli.py report --run results/run1
```

A search prints one line per generation and ends with a metrics block:
the winning genome, its screening mean, its mean on a DISJOINT confirm
panel, and the gap between the two, which is the winner's curse made
visible. Quote the confirm mean, never the screen mean.

## From search to submission

Searches feed a rotating top-N pool (shared across runs), the arena
ranks the pool head to head, and submit packages any member as a
single-file agent with a hard precheck battery:

```bash
# every search rotates its best genomes into the pool
python cli.py search --hours 2 --pool results/pool.json --pool-size 10
python cli.py pool                        # show the current top N

# all-vs-all in the real engine: BOTH seat orientations per seed,
# ranked by a Bradley-Terry fit on wins (the model the competition
# itself uses for final standings). Bank-vs-idle selects FOR the pool;
# the arena ranks WITHIN it, and the two disagree exactly when a
# schedule is rich but fragile on a shared market.
python cli.py arena --pool results/pool.json --seeds 11,23,47

# package pool member #1 as a self-contained stdlib-only main.py and
# precheck it: syntax, stdlib-only imports, a real smoke episode, a
# SEAT-1 episode on the raw trimmed observation, worst-turn latency
# against the 1 s budget, and determinism (same seed, same bank)
python cli.py submit --pool results/pool.json --rank 1 --out submission
```

`submit` refuses to bless anything that fails a check, and prints the
`kaggle competitions submit` line when everything passes. The seat-1
check earns its place: seats above 0 see a trimmed observation in local
replays, and an agent that assumes the full view dies there silently.

## Configuration

Everything is a parameter. Copy `config.example.json` and pass it with
`--config`:

| key | default | meaning |
|---|---|---|
| `species` | envelope, boundmix, intensity, random | one island each |
| `pop` | 8 | genomes per island |
| `seeds` | 11, 23, 47 | the CRN screen panel |
| `confirm_seeds` | 5, 13, 29, 61, 83 | disjoint final read |
| `migrate_every` | 6 | ring migration period, in generations |
| `tournament` | 3 | selection pressure |
| `crossover_p` | 0.6 | block-crossover probability |
| `hours` | 1.0 | wall-clock budget |
| `procs` | cpus-2 | parallel engine games |
| `rng_seed` | 20260824 | reproducibility of the search itself |
| `pool_file` | results/pool.json | the rotating top-N snapshot pool |
| `pool_size` | 10 | how many genomes the pool keeps |
| `islands` | 0 | island count; 0 = one per species, more cycles the list |
| `stagnation_gens` | 10 | island restart after N generations without improving |
| `global_stagnation_gens` | 18 | archipelago restart after M flat generations |
| `immigrant_every` | 4 | a fresh random genome replaces each island's worst |

## Escaping local optima

Long runs die of premature convergence: every island ends up holding
the same champion and generation after generation evaluates nothing new
(a 144-generation run of mine spent its last thirty generations flat).
Four mechanisms watch for it, all parameters above:

* **stagnation counter per island**: improving means beating the
  island's own best, not holding it (the elite always holds it);
* **island cataclysm**: after `stagnation_gens` flat generations the
  island keeps its elite and is reseeded around it with fresh species
  draws plus hot mutants (3-6 moves); a still-stuck island repeats this
  every N generations, which is intended: keep kicking;
* **hypermutation**: a half-stagnant island mutates hotter (2-5 moves
  instead of 1-3) before the cataclysm fires;
* **archipelago restart**: when the GLOBAL best is flat for
  `global_stagnation_gens`, only the island holding it survives and the
  rest are reborn from their species;
* plus **random immigrants** every `immigrant_every` generations as
  constant background diversity.

More islands: set `islands: 8` (the species list is cycled), or repeat
names in `species`. And one honest warning for long runs: the
winner's-curse gap grows with generations spent on a small screen
panel, because the search learns the panel. Give a long run a bigger
`seeds` list; the confirm panel will tell you if you did not.

The genome's own bounds (quadrant unlock windows, hire plateau, crop
tables) live in `islandga/genome.py` as `BOUNDS`, next to the four
species definitions. Editing a species IS the intended workflow: they
are priors, not truths.

## Use it as a library

The CLI is a thin wrapper. `pip install -e .` and everything is
importable, and `run_search` is EXECUTOR-AGNOSTIC: three injectable
seams let you search over your own executor while keeping the whole
orchestration (islands, escapes, pool rotation, the confirm
discipline).

```python
import json
from islandga.compiler import compile_spec
from islandga.search import SearchConfig, run_search

# the stock path, no CLI
cfg = SearchConfig.load("config.example.json")
cfg.hours = 2.0
metrics = run_search(cfg, "results/lib_run")
print(metrics["confirm_mean"])          # quote this one

# your own executor: implement the three seams
def my_compiler(genome):
    # "makeup" if YOUR executor retries structural purchases itself,
    # "reference" if it has valves like the bundled one (a re-emission
    # ladder under a valve-less executor double-buys)
    return compile_spec(genome, profile="makeup")

class InlinePool:                       # simplest seam: no multiprocessing
    def close(self): pass
    def join(self): pass

def my_eval_stream(pool, tasks):
    for gid, bp_json, seed in tasks:    # (gid, blueprint-json, seed)
        yield gid, seed, my_bank(json.loads(bp_json), seed)

metrics = run_search(cfg, "results/my_executor_run",
                     compiler=my_compiler,
                     pool_factory=lambda procs: InlinePool(),
                     eval_stream=my_eval_stream)
```

`my_bank` is the ~15-line harness in `islandga/evaluate.py`: build your
agent from the blueprint, play one episode against an idle opponent at
the given seed, return the final bank. The search neither knows nor
cares which executor produced the number; that is what makes the
package usable as a tool by a private research stack (it is exactly how
I use it).

## What to improve first

The reference executor is deliberately simple: greedy nearest-job
dispatch with a fixed priority stack and two safety valves. It exists
so the pipeline runs end to end, and it is the MULTIPLIER on every
schedule this search finds: better routing (day tours, en-route work,
sticky assignments) is worth tens of percent of bank before a single
gene changes. If you improve one file, improve `executor.py`.

Second: the sell policies in `compiler.py` (a gene, three variants)
moved final bank by double-digit percentages in my runs. The drainage
details of this engine matter more than intuition says: the shed caps
at 100 units, every deposit path respects the cap, and animal yield
arrives as a lump at first harvest.

## Engine facts this code leans on

All in `islandga/engine_facts.py`, copied verbatim from the engine
(1.32.7), because this project once re-derived a price branch from
memory and the bound came out 3.7x too high:

* prices: additive-amplitude scarcity/glut branches, integer-rounded,
  floor $1; carrot, tomato and egg carry a convex `hinge` above their
  knee;
* the n-th hire of a day costs fib(n): twelve hands are $376/day;
* market orders settle by list index (sells fund the purchases behind
  them) and every purchase fails silently and free when the pocket is
  short: schedule optimistically, repair at runtime;
* a plant left unwatered two days running dies the second night.

## Citing

If this repo or the notebook feeds something you publish, a link to
either is plenty:

* code: `github.com/destbreso/kaggriculture-island-ga`
* reasoning: [Island GA | An owned schedule is a moat](https://www.kaggle.com/code/destbreso/island-ga-an-owned-schedule-is-a-moat)

MIT licensed. Built by [destbreso](https://www.kaggle.com/destbreso)
during the Kaggriculture competition; the schedules my own runs return
stay private while the competition is live, and the notebook's section
8 explains why that is the honest trade.
