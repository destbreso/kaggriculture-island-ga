"""Engine constants and pricing for Kaggriculture 1.32.7.

Everything here is copied verbatim from the competition engine
(``kaggle_environments/envs/kaggriculture/kaggriculture.py``) and is
public information. Keeping it in one module is what lets the compiler
and the relaxed bound stay engine-exact instead of engine-approximate:
this project once re-derived the glut branch from memory and the bound
came out 3.7x too high.
"""
import math

MARKET_I0 = 10000
HINGE_GAIN = 8.0
PRICE_FLOOR = 1

MARKET_PARAMS = {
    "WHEAT":      {"base": 25,  "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base": 35,  "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base": 60,  "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base": 50,  "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")

CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

LAND_PRICES = (1000, 2000, 4000)     # NE, SW, SE in unlock order
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
SHED_CAPACITY = 100
MAX_ORDERS_PER_TURN = 10


def shape(func, x, T=None):
    """The engine's price-shape functions, verbatim."""
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price(item, inventory):
    """Unit price at a given market inventory. Mirrors ``market_price``."""
    p = MARKET_PARAMS[item]
    base, T = p["base"], p["T"]
    if inventory < MARKET_I0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape(f, T, T)
        val = base + amp * shape(f, MARKET_I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape(f, T, T)
        val = base - amp * shape(f, inventory - MARKET_I0, T)
    return max(PRICE_FLOOR, int(round(val)))


def fib_hire_cost(n_already_today):
    """Cost of the next hire when n have already been hired today:
    fib(n) starting 1, 1, 2, 3, 5. Twelve hands cost $376 a day."""
    a, b = 1, 1
    for _ in range(n_already_today):
        a, b = b, a + b
    return a


def day_hire_cost(n_hands):
    return sum(fib_hire_cost(k) for k in range(n_hands))
