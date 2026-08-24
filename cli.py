"""Command-line interface for the island-GA schedule search.

    python cli.py bound
    python cli.py smoke  [--species envelope] [--seed 11]
    python cli.py eval   --genome examples/genome_envelope.json [--seeds 11,23,47]
    python cli.py search [--config config.json] [--hours 1] [--procs 8] [--out results/run1]
    python cli.py report --run results/run1
"""
import argparse
import json
import random
import time
from pathlib import Path

from islandga import bound as bound_mod
from islandga.compiler import compile_spec
from islandga.evaluate import bank_of, eval_genome
from islandga.genome import SPECIES, gid_of, make_species
from islandga.search import SearchConfig, run_search


def cmd_bound(_args):
    return bound_mod.main()


def cmd_smoke(args):
    g = make_species(args.species, random.Random(args.seed))
    bp = compile_spec(json.loads(json.dumps(g)))
    n_orders = sum(len(v) for v in bp["market"].values())
    print(f"species {args.species}  gid {gid_of(g)}")
    print(f"compiled: {len(bp['days'])} days, {n_orders} market orders, "
          f"herd goal {bp['meta']['herd_goal']}")
    t0 = time.time()
    bank = bank_of(bp, args.seed)
    print(f"seed {args.seed}: bank ${bank:,.0f}  "
          f"({time.time()-t0:.1f}s/game)")
    return 0


def cmd_eval(args):
    genome = json.loads(Path(args.genome).read_text())
    seeds = [int(s) for s in args.seeds.split(",")]
    banks = eval_genome(genome, seeds)
    for s, b in banks.items():
        print(f"seed {s}: ${b:,.0f}")
    print(f"mean over {len(seeds)} seeds: "
          f"${sum(banks.values())/len(banks):,.0f}")
    return 0


def cmd_search(args):
    cfg = SearchConfig.load(args.config) if args.config else SearchConfig()
    if args.hours is not None:
        cfg.hours = args.hours
    if args.procs is not None:
        cfg.procs = args.procs
    out = args.out or f"results/run_{int(time.time())}"
    print(f"island GA: {len(cfg.species)} islands x {cfg.pop}, "
          f"screen seeds {cfg.seeds}, {cfg.hours}h -> {out}")
    run_search(cfg, out)
    return 0


def cmd_report(args):
    run = Path(args.run)
    best = json.loads((run / "best.json").read_text())
    rows = [json.loads(line) for line in (run / "log.jsonl").open()]
    by_gen = {}
    for r in rows:
        by_gen.setdefault(r["gen"], []).append(r["bank"])
    print(f"{'gen':>4s} {'games':>6s} {'gen mean':>10s} {'gen max':>10s}")
    for gen in sorted(by_gen):
        b = by_gen[gen]
        print(f"{gen:4d} {len(b):6d} {sum(b)/len(b):10,.0f} "
              f"{max(b):10,.0f}")
    print("\nbest genome:", best["best_gid"])
    print(f"screen mean  ${best['screen_mean']:,.0f}  "
          f"(seeds {sorted(best['screen_banks'])})")
    print(f"CONFIRM mean ${best['confirm_mean']:,.0f}  "
          f"(disjoint seeds {sorted(best['confirm_banks'])})")
    print(f"winner's-curse gap ${best['winners_curse_gap']:,.0f}  "
          f"<- why the confirm panel exists")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gens = sorted(by_gen)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(gens, [max(by_gen[g]) for g in gens], "-o", ms=3,
                color="#2f6fb2", label="best game of the generation")
        ax.plot(gens, [sum(by_gen[g]) / len(by_gen[g]) for g in gens],
                "-", color="#9aa3ad", label="generation mean")
        ax.set_xlabel("generation")
        ax.set_ylabel("bank vs idle, $")
        ax.legend(frameon=False, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(run / "trajectory.png", dpi=130)
        print(f"trajectory chart -> {run/'trajectory.png'}")
    except ImportError:
        print("(matplotlib not installed: skipping the trajectory chart)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bound", help="print the relaxed bound and its mix")

    p = sub.add_parser("smoke", help="compile one species, play one game")
    p.add_argument("--species", default="envelope", choices=sorted(SPECIES))
    p.add_argument("--seed", type=int, default=11)

    p = sub.add_parser("eval", help="evaluate a genome file on a seed panel")
    p.add_argument("--genome", required=True)
    p.add_argument("--seeds", default="11,23,47")

    p = sub.add_parser("search", help="run the island GA")
    p.add_argument("--config", default=None)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--procs", type=int, default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("report", help="summarise a finished run")
    p.add_argument("--run", required=True)

    args = ap.parse_args()
    return {"bound": cmd_bound, "smoke": cmd_smoke, "eval": cmd_eval,
            "search": cmd_search, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
