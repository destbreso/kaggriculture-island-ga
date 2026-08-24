"""The genome: a declarative season SPEC, never an action stream.

A genome is twelve numbers and four small tables. The compiler
(``compiler.py``) derives the whole market channel from it, which is
what makes mutation and block crossover produce coherent, financeable
schedules by construction. Splicing raw action streams instead breaks
the financing chain, because market orders settle by list index and
sells fund the purchases behind them.

The four SPECIES are grown from public priors, never from anyone's
recorded route: seeding a search with a transcribed strong schedule
loses on three counts (tournament takeover, transcription loss, and it
destroys the convergence-versus-niches measurement). The reasoning, with
a takeover simulation, is section 6.1 of the companion notebook.
"""
import copy
import hashlib
import json
import random

from .engine_facts import ANIMALS

QUADRANTS = ("NW", "NE", "SW", "SE")
CROP_FILL_ORDER = ("TOMATO", "STRAWBERRY", "CARROT", "WHEAT", "MELON")
SELL_POLICIES = ("daily", "sweep", "hybrid")

BOUNDS = {
    "ne": (4, 8),            # NE unlock day
    "sw": (7, 12),           # SW unlock day
    "se": (11, 14),          # SE unlock day when present (or None)
    "plateau": (8, 14),      # hands/day cap
    "ramp_full": (4, 12),    # days to reach the plateau
    "tomato_day": (8, 16),   # tomato window start
}


def gid_of(genome):
    return hashlib.sha1(json.dumps(genome, sort_keys=True)
                        .encode()).hexdigest()[:10]


# ---------------------------------------------------------------- species
def species_envelope():
    """Leader-envelope medians: aggregates measured across the top of the
    ladder (quadrant windows, hire plateaus, crop ranges), not anyone's
    route."""
    return {"ne": 6, "sw": 10, "se": None, "plateau": 12, "ramp_full": 8,
            "tomato_day": 8, "melon2": 1, "sellpol": "hybrid",
            "herd": [[0, "COW", 2], [0, "SHEEP", 2], [6, "COW", 2],
                     [8, "SHEEP", 1], [10, "COW", 1]],
            "prog": {"NW": {"WHEAT": 8, "CARROT": 3, "STRAWBERRY": 4,
                            "MELON": 2},
                     "NE": {"WHEAT": 10, "CARROT": 4, "STRAWBERRY": 5,
                            "MELON": 6},
                     "SW": {"WHEAT": 12, "CARROT": 5, "MELON": 8},
                     "SE": {}}}


def species_boundmix():
    """The relaxed bound's optimal mix leans: tomato window on, geese for
    the egg line."""
    g = species_envelope()
    g["herd"] = [[0, "COW", 2], [0, "SHEEP", 2], [4, "GOOSE", 4],
                 [8, "COW", 2]]
    g["prog"]["NW"] = {"WHEAT": 6, "CARROT": 2, "STRAWBERRY": 4,
                       "TOMATO": 4, "MELON": 1}
    g["prog"]["NE"] = {"WHEAT": 8, "CARROT": 3, "STRAWBERRY": 4,
                       "TOMATO": 4, "MELON": 6}
    return g


def species_intensity():
    """Fourth quadrant unlocked, maximum hire plateau."""
    g = species_envelope()
    g.update(se=12, plateau=14, ramp_full=6)
    g["herd"] = g["herd"] + [[12, "SHEEP", 2]]
    g["prog"]["SE"] = {"WHEAT": 10, "MELON": 10, "CARROT": 5}
    return g


def species_random(rng):
    """Uniform within the envelope bounds."""
    g = {"ne": rng.randint(*BOUNDS["ne"]), "sw": rng.randint(*BOUNDS["sw"]),
         "se": rng.choice([None, None, 11, 12, 13]),
         "plateau": rng.randint(*BOUNDS["plateau"]),
         "ramp_full": rng.randint(*BOUNDS["ramp_full"]),
         "tomato_day": rng.randint(*BOUNDS["tomato_day"]),
         "melon2": rng.randint(0, 1),
         "sellpol": rng.choice(SELL_POLICIES), "herd": [], "prog": {}}
    for kind in ANIMALS:
        n = rng.randint(0, ANIMALS[kind]["max_held"])
        if n:
            g["herd"].append([rng.randint(0, 10), kind, n])
    for q in QUADRANTS:
        budget = 25 if (q != "SE" or g["se"]) else 0
        alloc = {}
        for crop in CROP_FILL_ORDER:
            n = rng.randint(0, max(0, budget))
            n = min(n, budget - sum(alloc.values()))
            if n:
                alloc[crop] = n
        g["prog"][q] = alloc
    return g


SPECIES = {"envelope": species_envelope, "boundmix": species_boundmix,
           "intensity": species_intensity, "random": species_random}


def make_species(name, rng):
    fn = SPECIES[name]
    return fn(rng) if name == "random" else fn()


# ---------------------------------------------------------------- moves
def mutate(genome, rng):
    """1-3 macro-mutations; every move keeps the spec meaningful."""
    g = copy.deepcopy(genome)
    for _ in range(rng.randint(1, 3)):
        m = rng.randrange(7)
        if m == 0:
            k = rng.choice(["ne", "sw", "se"])
            if k == "se":
                g["se"] = rng.choice([11, 12, 13]) if g["se"] is None else \
                    (None if rng.random() < 0.5
                     else max(BOUNDS["se"][0],
                              min(BOUNDS["se"][1],
                                  g["se"] + rng.choice([-1, 1]))))
            else:
                lo, hi = BOUNDS[k]
                g[k] = max(lo, min(hi, g[k] + rng.choice([-1, 1])))
        elif m == 1:
            lo, hi = BOUNDS["plateau"]
            g["plateau"] = max(lo, min(hi, g["plateau"]
                                       + rng.choice([-1, 1])))
            lo, hi = BOUNDS["ramp_full"]
            g["ramp_full"] = max(lo, min(hi, g["ramp_full"]
                                         + rng.choice([-2, 2])))
        elif m == 2 and g["herd"]:
            w = rng.choice(g["herd"])
            f = rng.random()
            if f < 0.4:
                w[2] = max(0, w[2] + rng.choice([-1, 1]))
                g["herd"] = [x for x in g["herd"] if x[2] > 0]
            elif f < 0.8:
                w[0] = max(0, min(16, w[0] + rng.choice([-2, 2])))
            else:
                g["herd"].append([rng.randint(0, 12),
                                  rng.choice(list(ANIMALS)), 1])
        elif m == 3:
            q = rng.choice(list(QUADRANTS))
            alloc = dict(g["prog"].get(q) or {})
            crops = list(CROP_FILL_ORDER) + ["FALLOW"]
            src = rng.choice([c for c in crops
                              if alloc.get(c, 0) > 0] or ["FALLOW"])
            dst = rng.choice([c for c in crops if c != src])
            k = rng.randint(1, 2)
            if src != "FALLOW":
                k = min(k, alloc.get(src, 0))
                alloc[src] = alloc.get(src, 0) - k
            if dst != "FALLOW" and k:
                if sum(alloc.values()) + k <= 25:
                    alloc[dst] = alloc.get(dst, 0) + k
            g["prog"][q] = {c: n for c, n in alloc.items() if n > 0}
        elif m == 4:
            lo, hi = BOUNDS["tomato_day"]
            g["tomato_day"] = max(lo, min(hi, g["tomato_day"]
                                          + rng.choice([-2, 2])))
        elif m == 5:
            g["melon2"] = 1 - g.get("melon2", 1)
        else:
            g["sellpol"] = rng.choice(list(SELL_POLICIES))
    return g


def crossover(a, b, rng):
    """Block crossover: whole subsystems travel together, because crops,
    labour and cash are co-adapted through the shared hands and the
    shared pocket."""
    child = {}
    src = a if rng.random() < 0.5 else b
    for k in ("ne", "sw", "se"):
        child[k] = src[k]
    src = a if rng.random() < 0.5 else b
    child["plateau"], child["ramp_full"] = src["plateau"], src["ramp_full"]
    child["herd"] = copy.deepcopy((a if rng.random() < 0.5 else b)["herd"])
    src = a if rng.random() < 0.5 else b
    child["tomato_day"] = src["tomato_day"]
    child["melon2"] = src.get("melon2", 1)
    child["sellpol"] = (a if rng.random() < 0.5 else b).get("sellpol",
                                                            "hybrid")
    child["prog"] = {q: copy.deepcopy(
        (a if rng.random() < 0.5 else b)["prog"].get(q, {}))
        for q in QUADRANTS}
    return child
