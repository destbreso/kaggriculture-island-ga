"""The compiler: a genome SPEC in, a playable blueprint out.

``compile_spec(genome)`` returns ``{"days", "market", "meta"}``:
per-day construction targets, the complete market channel with financing
derived from scratch, and bookkeeping the executor's repair valve reads.

Five derivations, in order (the contract is section 5 of the companion
notebook):

1. tile realisation      structures nearest the shed, crops fill the rest
2. plant-day unrolling   cycles, waves and windows from the engine table
3. production projector  cumulative sellable-by-morning series per good
4. market channel        sells at hour 0, purchases then hires at hour 1+
5. financing by optimism orders at the spec's own days, never at days a
                         cash model approves

Point 5 is a finding, not a preference. Every purchase in this engine
fails SILENTLY and at zero cost when the pocket is short, while a plant
left unwatered two days running dies the second night: scheduling too
little is lethal and scheduling too much is free. Cautious compile-time
treasurers kept starving real farms; the engine's own refusals are the
brake, plus the executor's small repair valve for structural purchases.
"""
from .engine_facts import ANIMALS, CROPS, SHED_TILES

QUAD_ORIGIN = {"NW": (0, 0), "NE": (5, 0), "SW": (0, 5), "SE": (5, 5)}
CROP_FILL_ORDER = ("TOMATO", "STRAWBERRY", "CARROT", "WHEAT", "MELON")
EFF = 0.9                              # executor imperfection on yields


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quad_tiles(q):
    x0, y0 = QUAD_ORIGIN[q]
    tiles = [(x0 + i, y0 + j) for j in range(5) for i in range(5)]
    return sorted(tiles, key=lambda t: (min(_dist(t, s) for s in SHED_TILES),
                                        t))


def compile_spec(g, profile="reference"):
    """Compile a genome. `profile` declares which REPAIR LAYER the
    executor provides, because structural-purchase emission must match it:

    * "reference": the bundled executor (animal ledger valve). Animal
      waves re-emit on a ladder (d, +2, +4, +7, +10) and the valve makes
      re-emission safe; a briefly-poor farm keeps its herd.
    * "makeup": an executor that RETRIES structural purchases itself
      (windowed land re-issue, per-species animal makeup). Animal waves
      emit ONCE; a ladder under such an executor would double-buy.

    Land re-emits on a short ladder under both profiles; note the known
    softness this buys: an executor without a land gate can clear a
    later re-emission early, so unlock genes are intentions, not exact
    days, and the search optimises what actually happens."""
    unlock = {"NW": 0, "NE": int(g["ne"]), "SW": int(g["sw"]),
              "SE": int(g["se"]) if g.get("se") else 99}

    # herd waves clamped to the engine's per-species caps
    waves = []
    held = {k: 0 for k in ANIMALS}
    for day, kind, n in sorted(g["herd"], key=lambda w: w[0]):
        n = min(int(n), ANIMALS[kind]["max_held"] - held[kind])
        if n > 0:
            held[kind] += n
            waves.append([int(day), kind, n])

    # ---- 1) tile realisation
    free = {q: quad_tiles(q) for q in QUAD_ORIGIN}
    placed = []                        # (x, y, kind, struct, day)
    for day, kind, n in waves:
        for _ in range(n):
            opts = [(min(_dist(t, s) for s in SHED_TILES), q, t)
                    for q in QUAD_ORIGIN if unlock[q] <= day and free[q]
                    for t in (free[q][0],)]
            if not opts:
                break
            _, q, t = min(opts)
            free[q].pop(0)
            placed.append((t[0], t[1], kind, ANIMALS[kind]["structure"], day))

    crop_tiles = []                    # (x, y, crop, active_day)
    for q in QUAD_ORIGIN:
        if unlock[q] > 29:
            continue
        want = dict(g["prog"].get(q) or {})
        for crop in CROP_FILL_ORDER:
            n = int(want.get(crop, 0))
            while n > 0 and free[q]:
                t = free[q].pop(0)
                crop_tiles.append((t[0], t[1], crop, unlock[q]))
                n -= 1

    # ---- 2) plant-day unrolling
    tom_d = int(g.get("tomato_day", 8))
    plant_days = []                    # (x, y, crop, [plant days])
    for x, y, crop, a in crop_tiles:
        c = CROPS[crop]
        m = c["max_yield_day"]
        if crop in ("WHEAT", "CARROT"):
            days = list(range(a, 30 - m, m + 1))
        elif crop == "MELON":
            days = [a] + ([a + m + 1] if g.get("melon2", 1)
                          and a + 2 * m + 1 <= 29 else [])
        elif crop == "STRAWBERRY":
            if a > 12:                 # too late for the ongoing window
                crop, c, m = "WHEAT", CROPS["WHEAT"], 4
                days = list(range(a, 30 - m, m + 1))
            else:
                days = [a]
        else:                          # TOMATO
            d0 = max(tom_d, a)
            days = [d0] if d0 <= 18 else []
        if days:
            plant_days.append((x, y, crop, days))

    # per-day plant targets: cyclers stay listed through their span so the
    # executor replants whenever the tile empties; waves get +-2 slack
    day_plants = {d: [] for d in range(30)}
    for x, y, crop, days in plant_days:
        if crop in ("WHEAT", "CARROT"):
            m = CROPS[crop]["max_yield_day"]
            for d in range(days[0], min(29, days[-1] + m) + 1):
                day_plants[d].append([x, y, crop])
        else:
            for p in days:
                for d in range(p, min(29, p + 2) + 1):
                    day_plants[d].append([x, y, crop])

    day_animals = {d: [] for d in range(30)}
    day_builds = {d: [] for d in range(30)}
    for x, y, kind, struct, b in placed:
        for d in range(max(0, b - 1), 30):
            day_builds[d].append([x, y, struct])
        for d in range(b, 30):
            day_animals[d].append([x, y, kind, struct])

    seed_buys = {d: {} for d in range(30)}
    for x, y, crop, days in plant_days:
        for i, p in enumerate(days):
            d = p if i == 0 else max(0, p - 1)
            seed_buys[d][crop] = seed_buys[d].get(crop, 0) + 1

    # workload per day, for the hire cap (~5 work-tiles per hand)
    active_tiles = [0] * 30
    for x, y, crop, days in plant_days:
        for d in range(days[0], 30):
            active_tiles[d] += 1
    for x, y, kind, struct, b in placed:
        for d in range(b, 30):
            active_tiles[d] += 2

    hires = [0] * 30
    plateau, ramp = int(g["plateau"]), max(1, int(g["ramp_full"]))
    for d in range(30):
        want = min(plateau, 2 + (plateau * d) // ramp,
                   2 + active_tiles[d] // 5)
        if active_tiles[d] > 8:
            want = max(want, 2)
        hires[d] = want

    # ---- 3) production projector: cumulative sellable-by-morning.
    # Built as per-day arrivals first, then prefix-summed: crops arrive
    # per harvest event (ongoing crops STEP at their interval, matching
    # the engine's yield_units, not a smoothed drip).
    prod = {d: {} for d in range(31)}

    def add(day, good, units):
        if day <= 30:
            prod[max(0, day)][good] = \
                prod[max(0, day)].get(good, 0.0) + units

    for x, y, crop, days in plant_days:
        c = CROPS[crop]
        if not c["ongoing"]:
            for p in days:
                add(p + c["max_yield_day"] + 1, crop,
                    c["max_yield"] * EFF)
        else:
            p = days[0]
            for d in range(p + c["first_yield_day"], 30, c["interval"]):
                add(d + 1, crop, 1 * EFF)

    cum = {}
    for d in range(31):
        for good, u in prod.get(d, {}).items():
            arr = cum.setdefault(good, [0.0] * 31)
            for k in range(d, 31):
                arr[k] += u

    herd_by_day = [0] * 31
    for day0, kind, n in waves:
        a = ANIMALS[kind]
        arr = cum.setdefault(a["product"], [0.0] * 31)
        # yield ACCRUES from placement but is harvestable only from the
        # species' first-yield day, so the first harvest is a lump
        for d in range(day0 + a["first_yield_day"], 31):
            arr[d] += n * (d - 1 - day0) / a["interval"]
        for d in range(day0, 30):
            herd_by_day[d + 1] += n
        fert = cum.setdefault("FERTILIZER", [0.0] * 31)
        for d in range(day0 + 2, 31):
            fert[d] += (d - 1 - day0) * n * 0.4

    # ---- 4) the market channel
    market = {}
    swept = {}
    pol = g.get("sellpol", "hybrid")

    def push(turn, order):
        market.setdefault(str(turn), [])
        if len(market[str(turn)]) < 10:
            market[str(turn)].append(order)
            return True
        return False

    for d in range(30):
        t0, t1, t2 = d * 24, d * 24 + 1, d * 24 + 2
        # sells at hour 0. SELL settles per unit and stops at an empty
        # shed, so a modest overshoot self-clamps; but sells are also
        # demand signals to an executor that harvests to order, so the
        # sizing stays NEAR-exact. Three policies, chosen by the gene:
        if d >= 2:
            feed_cum = sum(herd_by_day[k] for k in range(d))
            for good in sorted(cum):
                delta = cum[good][d] - cum[good][d - 1]
                if pol == "daily":
                    q = delta * 1.25 + (3 if delta > 0.5 else 0)
                    if good == "WHEAT":
                        q = q - herd_by_day[d] - 1
                elif pol == "sweep":
                    target = cum[good][d] * 1.08 + 2
                    if good == "WHEAT":
                        target -= feed_cum
                    q = target - swept.get(good, 0)
                else:                  # hybrid: daily pace + 3-day top-up
                    q = delta * 1.15 + (2 if delta > 0.5 else 0)
                    if good == "WHEAT":
                        q = q - herd_by_day[d]
                    if d % 3 == 0:
                        target = cum[good][d] * 1.05 + 2
                        if good == "WHEAT":
                            target -= feed_cum
                        q = max(q, target - swept.get(good, 0))
                q = int(q)
                if q >= 1:
                    push(t0, ["SELL", good, q])
                    swept[good] = swept.get(good, 0) + q
        if d >= 28:                    # terminal flush, phantom by design
            for good in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                         "EGG", "MILK", "WOOL", "FERTILIZER"):
                push(t0 if d == 29 else t1, ["SELL", good, 999])
        if d == 29:                    # and the last evening's harvests
            for gi, good in enumerate(("WHEAT", "CARROT", "TOMATO",
                                       "STRAWBERRY", "MELON", "EGG", "MILK",
                                       "WOOL", "FERTILIZER")):
                push(29 * 24 + 21 + gi // 5, ["SELL", good, 999])
        # purchases at hour 1, seeds strictly before land
        for crop in sorted(seed_buys[d]):
            n = seed_buys[d][crop]
            if n > 0:
                push(t1, ["BUY_SEED", crop, n + 1])
        if herd_by_day[d] > 0 and d < 28:
            push(t1, ["BUY_PRODUCT", "WHEAT", herd_by_day[d]])
        # structural purchases: emission profile per the docstring
        animal_days = ((0, 2, 4, 7, 10) if profile == "reference"
                       else (0,))
        for day0, kind, n in waves:
            if any(d == day0 + k for k in animal_days) and d <= 29:
                push(t1, ["BUY_ANIMAL", kind, n])
        for i, q in enumerate(("NE", "SW", "SE")):
            if unlock[q] <= 29 and d in (unlock[q], unlock[q] + 2,
                                         unlock[q] + 4, unlock[q] + 6):
                push(t1, ["BUY_LAND"])
        # the day's hands: unconditional at the curve+size cap (a failed
        # HIRE is free; an unscheduled hand kills plants)
        for h in range(hires[d]):
            if not push(t1, ["HIRE"]):
                push(t2, ["HIRE"])

    days = {str(d): {"plants": day_plants[d], "animals": day_animals[d],
                     "buildings": day_builds[d]} for d in range(30)}
    meta = {"herd_goal": {k: held[k] for k in held if held[k]},
            "unlock": unlock, "sellpol": pol}
    return {"days": days, "market": market, "meta": meta}
