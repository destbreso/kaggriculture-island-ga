"""The reference executor: a compiled blueprint in, per-turn actions out.

Deliberately SIMPLE: greedy nearest-job dispatch under a fixed priority
stack, with per-turn claims so two workers never chase the same tile,
plus two small valves:

* delivery-before-sells: a worker carrying a good the schedule sells in
  the next 8 turns walks it to the shed first, because a sell that finds
  an empty shed silently shorts and de-funds the purchase behind it;
* the animal ledger: the compiler re-emits BUY_ANIMAL orders on a short
  ladder so a briefly-poor farm does not lose its herd, and this valve
  drops re-emissions once the species' goal is met, so re-emission can
  never double-buy.

This executor exists so the pipeline runs end to end. It is the
MULTIPLIER on every schedule, and it is the part you should improve:
smarter routing alone is worth tens of percent of bank.
"""
from .engine_facts import ANIMALS, CROPS, SHED_TILES


def _get(v, k, d=None):
    if isinstance(v, dict):
        return v.get(k, d)
    return getattr(v, k, d)


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _toward(pos, tgt):
    x, y = pos
    tx, ty = tgt
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _shed_gate(pos):
    return min(SHED_TILES, key=lambda s: _dist(pos, s))


# priority stack: smaller = more urgent. WATER sits WITH harvest, above
# planting and building: the first version put it below them, planted
# new tiles while the standing ones died (17 plants to ZERO by day 9),
# and the whole season starved. Water and harvest at the same rank mix
# by distance, which is what a small crew needs.
P_URGENT_WATER, P_FEED, P_HARVEST, P_WATER = 0, 1, 2, 2
P_PLANT, P_BUILD, P_COLLECT, P_CARE, P_DIG = 3, 4, 5, 6, 8


def water_prio(hour):
    return P_WATER


class ReferenceExecutor:
    def __init__(self, bp):
        self.bp = bp
        self.goal = dict((bp.get("meta") or {}).get("herd_goal") or {})

    # ------------------------------------------------------------ helpers
    def _market_orders(self, turn, grid, shed, carried):
        """The scheduled orders, through the animal ledger valve."""
        owned = {}
        for row in grid:
            for tl in row:
                if isinstance(tl, dict) and tl.get("animal"):
                    k = tl["animal"]
                    k = k.get("kind") if isinstance(k, dict) else k
                    owned[k] = owned.get(k, 0) + 1
        out = []
        for o in (self.bp["market"].get(str(turn)) or []):
            if o and o[0] == "BUY_ANIMAL" and len(o) > 2:
                kind = o[1]
                have = (owned.get(kind, 0) + int(shed.get(kind, 0) or 0)
                        + carried.get(kind, 0))
                room = self.goal.get(kind, 0) - have
                if room <= 0:
                    continue
                out.append(["BUY_ANIMAL", kind, min(int(o[2]), room)])
            else:
                out.append(list(o))
        return out[:10]

    def _jobs(self, day, hour, grid, target, seeds):
        want_plant = {(x, y): crop
                      for x, y, crop in (target.get("plants") or [])}
        want_animal = {(x, y): (kind, struct) for x, y, kind, struct
                       in (target.get("animals") or [])}
        for x, y, struct in (target.get("buildings") or []):
            want_animal.setdefault((x, y), (None, struct))

        jobs = []                      # (prio, x, y, verb)
        budget = dict(seeds)
        for y, row in enumerate(grid):
            for x, tl in enumerate(row):
                pos = (x, y)
                if tl is None:
                    crop = want_plant.get(pos)
                    if crop and budget.get(crop, 0) > 0:
                        budget[crop] -= 1
                        jobs.append((P_PLANT, x, y, ["PLANT", crop]))
                    elif pos in want_animal:
                        struct = want_animal[pos][1]
                        verb = "BUILD_COOP" if struct == "COOP" \
                            else "BUILD_PASTURE"
                        jobs.append((P_BUILD, x, y, [verb]))
                    continue
                if not isinstance(tl, dict):
                    continue
                kind = tl.get("kind")
                if kind == "PLANT":
                    crop = tl.get("crop")
                    c = CROPS.get(crop) or {}
                    pd = tl.get("planted_day")
                    age = day - (day if pd is None else int(pd))
                    if not tl.get("watered_today"):
                        urgent = int(tl.get("consecutive_unwatered", 0)
                                     or 0) >= 1
                        jobs.append((P_URGENT_WATER if urgent
                                     else water_prio(hour),
                                     x, y, ["WATER"]))
                    yu = int(tl.get("yield_units", 0) or 0)
                    myd = c.get("max_yield_day", 99)
                    ripe = (yu >= c.get("max_yield", 99) or age > myd
                            or (age == myd and tl.get("watered_today")))
                    can = age >= c.get("first_yield_day", 0)
                    if yu > 0 and can and (c.get("ongoing") or ripe):
                        jobs.append((P_HARVEST, x, y, ["HARVEST"]))
                    if (yu == 0 and not c.get("ongoing") and age > myd):
                        jobs.append((P_DIG, x, y, ["DIG"]))
                elif tl.get("animal"):
                    if not tl.get("fed_today"):
                        jobs.append((P_FEED, x, y, ["FEED"]))
                    if not tl.get("cared_today"):
                        jobs.append((P_CARE, x, y, ["CARE"]))
                    if tl.get("fertilizer_available"):
                        jobs.append((P_COLLECT, x, y,
                                     ["COLLECT_FERTILIZER"]))
                    if int(tl.get("yield_units", 0) or 0) > 0:
                        jobs.append((P_HARVEST, x, y, ["HARVEST"]))
                elif kind in ("PASTURE", "COOP"):
                    w = want_animal.get(pos)
                    if w and w[0]:
                        jobs.append((P_BUILD, x, y, ["PLACE", w[0]]))
                elif kind == "WEED":
                    jobs.append((P_DIG, x, y, ["DIG"]))
        return jobs

    def _upcoming_sells(self, turn, horizon=8):
        need = {}
        for t in range(turn, turn + horizon):
            for o in (self.bp["market"].get(str(t)) or []):
                if o and o[0] == "SELL" and len(o) > 2:
                    need[o[1]] = need.get(o[1], 0) + int(o[2] or 0)
        return need

    # ------------------------------------------------------------ the turn
    def act(self, obs):
        seat = 0
        for k in ("player_index", "index", "player"):
            s = _get(obs, k)
            if s is not None:
                seat = int(s)
                break
        day = int(_get(obs, "day", 0) or 0)
        hour = int(_get(obs, "hour", 0) or 0)
        turn = day * 24 + hour

        farms = list(_get(obs, "farms", []) or [])
        farm = farms[seat] if len(farms) > seat else {}
        grid = list(_get(farm, "tiles", []) or [])
        private = _get(obs, "private", {}) or {}
        seeds = dict(_get(private, "seeds", {}) or {})
        shed = dict(_get(private, "shed", {}) or {})
        invs = list(_get(private, "inventories", []) or [])
        units = [_get(farm, "farmer", None)] + list(_get(farm, "hands", [])
                                                    or [])
        carried = {}
        for iv in invs:
            if isinstance(iv, dict):
                for g, n in iv.items():
                    carried[g] = carried.get(g, 0) + int(n or 0)

        market = self._market_orders(turn, grid, shed, carried)
        target = self.bp["days"].get(str(min(day, 29))) or {}
        jobs = self._jobs(day, hour, grid, target, seeds)

        # seed makeup valve: plant deaths consume plantings the compiler
        # never scheduled seeds for; re-buy what the targets want and the
        # pouch cannot cover (once per day, at hour 3, capped)
        if hour == 3 and day < 26 and len(market) < 10:
            money = float(_get(farm, "money", 0) or 0)
            want = {}
            for x, y, crop in (target.get("plants") or []):
                if y < len(grid) and x < len(grid[y] or []) \
                        and grid[y][x] is None:
                    want[crop] = want.get(crop, 0) + 1
            for crop, n in sorted(want.items()):
                miss = n - int(seeds.get(crop, 0) or 0)
                cost = CROPS[crop]["seed"]
                if miss > 0 and money >= miss * cost + 300:
                    market.append(["BUY_SEED", crop, min(miss, 6)])
                    money -= min(miss, 6) * cost
            market = market[:10]
        by_pos = {}
        for prio, x, y, verb in jobs:
            by_pos.setdefault((x, y), []).append((prio, verb))
        sells = self._upcoming_sells(turn)
        short = {g: n - int(shed.get(g, 0) or 0) for g, n in sells.items()
                 if n > int(shed.get(g, 0) or 0)}
        if day >= 28:                  # terminal: hands empty into the shed
            for g, n in carried.items():
                if n > 0:
                    short[g] = max(short.get(g, 0), n)

        unfed = [(x, y) for prio, x, y, v in jobs if v == ["FEED"]]
        claimed = set()
        verbs = []
        for i, u in enumerate(units):
            if not u:
                verbs.append(["PASS"])
                continue
            pos = (int(u[0]), int(u[1]))
            inv = dict(invs[i]) if i < len(invs) and \
                isinstance(invs[i], dict) else {}
            wheat = int(inv.get("WHEAT", 0) or 0)
            fert = int(inv.get("FERTILIZER", 0) or 0)
            load = sum(int(n or 0) for n in inv.values())

            # 1) deliver ahead of the schedule's sells
            if short and any(int(inv.get(g, 0) or 0) > 0 for g in short):
                for g in list(short):
                    have = int(inv.get(g, 0) or 0)
                    if have > 0:
                        short[g] -= have
                        if short[g] <= 0:
                            del short[g]
                verbs.append(["DROP"] if pos in SHED_TILES
                             else _toward(pos, _shed_gate(pos)))
                continue

            # 2) a job under the feet costs zero travel
            here = None
            for prio, verb in sorted(by_pos.get(pos, ())):
                key = (verb[0], pos[0], pos[1])
                if key in claimed:
                    continue
                if verb == ["FEED"] and wheat <= 0:
                    continue
                # a PLACE without the animal in hand is an infinite loop:
                # the engine rejects it silently, the tile stays wanting,
                # and the unit repeats it forever (measured: 61-83 dead
                # PLACE actions a day before this guard existed)
                if verb[0] == "PLACE" and \
                        int(inv.get(verb[1], 0) or 0) <= 0:
                    continue
                here = (key, verb)
                break
            # opportunistic fertilize: doubles the watering yield inside
            # a non-ongoing crop's window
            if here is None and fert > 0:
                tl = grid[pos[1]][pos[0]] if pos[1] < len(grid) and \
                    pos[0] < len(grid[0] if grid else []) else None
                if isinstance(tl, dict) and tl.get("kind") == "PLANT":
                    c = CROPS.get(tl.get("crop")) or {}
                    pd = tl.get("planted_day")
                    age = day - (day if pd is None else int(pd))
                    myd = c.get("max_yield_day", 99)
                    if (not c.get("ongoing") and (myd + 1) // 2 <= age <= myd
                            and int(tl.get("fertilized_until_day", -1)
                                    or -1) < day):
                        here = (("FERTILIZE", pos[0], pos[1]), ["FERTILIZE"])
            if here:
                claimed.add(here[0])
                verbs.append(list(here[1]))
                continue

            # 3) fetch wheat when the herd is hungry and nobody carries
            if unfed and wheat <= 0 and int(shed.get("WHEAT", 0) or 0) > 0 \
                    and not any(int((invs[j] or {}).get("WHEAT", 0) or 0) > 0
                                for j in range(len(units)) if j < len(invs)):
                n = min(len(unfed), int(shed.get("WHEAT", 0) or 0), 6)
                verbs.append(["PICKUP", "WHEAT", n] if pos in SHED_TILES
                             else _toward(pos, _shed_gate(pos)))
                continue

            # 4) nearest job by (priority, distance), with claims
            best = None
            for prio, x, y, verb in jobs:
                key = (verb[0], x, y)
                if key in claimed:
                    continue
                if verb == ["FEED"] and wheat <= 0:
                    continue
                if verb[0] == "PLACE":
                    kind = verb[1]
                    if int(inv.get(kind, 0) or 0) <= 0:
                        # fetch the animal from the shed first
                        if int(shed.get(kind, 0) or 0) > 0 and \
                                ("FETCH", kind) not in claimed:
                            rank = (P_BUILD, _dist(pos, _shed_gate(pos)))
                            if best is None or rank < best[0]:
                                best = (rank, ("FETCH", kind),
                                        ["PICKUP", kind, 1],
                                        _shed_gate(pos))
                        continue
                rank = (prio, _dist(pos, (x, y)))
                if best is None or rank < best[0]:
                    best = (rank, key, verb, (x, y))
            if best is not None:
                _rank, key, verb, tgt = best
                claimed.add(key)
                verbs.append(list(verb) if pos == tgt
                             else _toward(pos, tgt))
                continue

            # 5) unload when heavy or late
            if load >= 6 or (load > 0 and hour >= 20):
                verbs.append(["DROP"] if pos in SHED_TILES
                             else _toward(pos, _shed_gate(pos)))
            else:
                verbs.append(["PASS"])

        return {"farmer": verbs[0] if verbs else ["PASS"],
                "hands": verbs[1:], "market": market}


def make_agent(bp):
    """A fresh agent callable for one episode."""
    ex = ReferenceExecutor(bp)
    return ex.act
