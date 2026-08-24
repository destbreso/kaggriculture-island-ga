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

The genome's own bounds (quadrant unlock windows, hire plateau, crop
tables) live in `islandga/genome.py` as `BOUNDS`, next to the four
species definitions. Editing a species IS the intended workflow: they
are priors, not truths.

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
