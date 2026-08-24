"""The crude relaxed bound: price the corridor BEFORE searching.

Revenue is the engine-exact end-of-season sale integral over the
verified 1.32.7 curves at median demand; tile-days are allocated
greedily by marginal revenue; animal products at engine caps; sells
trimmed at a $2 marginal. Everything ignored (walking, watering, action
budgets, financing timing) only LOWERS the truth, and costs are
deliberately under-estimated: both keep it a valid UPPER bound.
"""
from .engine_facts import MARKET_I0, price

DEMAND = {"WHEAT": 504, "STRAWBERRY": 396, "MILK": 288, "CARROT": 270,
          "TOMATO": 180, "WOOL": 180, "EGG": 180, "MELON": 30,
          "FERTILIZER": 0}
CROP = {"WHEAT": (2, 4, 6, False), "CARROT": (2, 3, 4, False),
        "TOMATO": (8, 8, 4, True), "STRAWBERRY": (10, 10, 4, True),
        "MELON": (10, 12, 6, False)}
TILE_DAYS = 60 * 28
ANIM_CAP = {"EGG": 4 * 27, "MILK": 6 * 12, "WOOL": 6 * 9}
FERT_UNITS = 16 * 29
COSTS = 60 * 10 + 5 * 400 + 6 * 500 + 4 * 300 + 1000 + 2000 + 300 * 20


def R(g, D, S):
    return sum(price(g, MARKET_I0 - D + i) for i in range(int(S)))


def per_tile_day(c):
    fy, my, y, ongoing = CROP[c]
    return (y / my) if not ongoing else (y + (30 - fy)) / 30


def compute():
    alloc = {c: 0 for c in CROP}
    units = {g: 0.0 for g in DEMAND}
    used, step = 0, 14
    while used + step <= TILE_DAYS:
        best, best_gain, best_add = None, 1.0, 0
        for c in CROP:
            add = per_tile_day(c) * step
            gain = R(c, DEMAND[c], units[c] + add) \
                - R(c, DEMAND[c], units[c])
            if gain > best_gain:
                best, best_gain, best_add = c, gain, add
        if not best:
            break
        units[best] += best_add
        alloc[best] += step
        used += step

    rows, gross = [], 0
    for g in DEMAND:
        S = units.get(g, 0.0)
        if g in ANIM_CAP:
            S = ANIM_CAP[g]
        if g == "FERTILIZER":
            S = FERT_UNITS
        S = int(S)
        while S > 0 and price(g, MARKET_I0 - DEMAND[g] + S - 1) <= 2:
            S -= 1
        r = R(g, DEMAND[g], S)
        rows.append((g, S, int(r)))
        gross += r
    rows.sort(key=lambda t: -t[2])
    return rows, alloc, int(3000 + gross - COSTS)


def main():
    rows, alloc, bound = compute()
    print(f"{'good':11s} {'units':>6s} {'revenue':>9s}")
    for g, S, r in rows:
        print(f"{g:11s} {S:6d} {r:9,d}")
    print(f"\ntile-day allocation: {alloc}")
    print(f"CRUDE RELAXED BOUND vs idle opponent: ${bound:,}")
    return 0
