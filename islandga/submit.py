"""Package a genome as a single-file agent and precheck it hard.

The built ``main.py`` is self-contained and stdlib-only: the engine
constants and the reference executor are concatenated FROM THEIR SOURCE
FILES (no template to drift out of sync), and the compiled blueprint is
embedded zlib+base85. A footer exposes ``agent(obs)``.

The precheck battery exists because a ladder crash is silent and a
submission slot is expensive:

1. syntax        the built file compiles
2. stdlib-only   every import is standard library
3. smoke         one real episode vs idle, bank must clear the idle floor
4. seat 1        one episode FROM SEAT 1 on the RAW (trimmed) observation,
                 because seat 1 sees less than seat 0 in local replays and
                 an agent that assumes the full view dies there
5. latency       worst per-turn wall time under the 1 s/turn budget
6. determinism   the same seed twice must produce the same bank
"""
import ast
import base64
import json
import re
import time
import zlib
from pathlib import Path

from .compiler import compile_spec

PKG = Path(__file__).parent
ALLOWED_IMPORTS = {"math", "json", "zlib", "base64"}
PASS_ACT = {"farmer": ["PASS"], "hands": [], "market": []}


def build_submission(genome, out_dir, note=""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bp = compile_spec(json.loads(json.dumps(genome)))
    blob = base64.b85encode(zlib.compress(
        json.dumps(bp, separators=(",", ":")).encode(), 9)).decode()

    facts = (PKG / "engine_facts.py").read_text()
    execu = (PKG / "executor.py").read_text()
    execu = re.sub(r"^from \.engine_facts import.*$", "", execu,
                   flags=re.M)
    header = ('"""Kaggriculture agent: an owned schedule found by island-GA'
              ' search.\n\nBuilt with github.com/destbreso/'
              'kaggriculture-island-ga (MIT).\n'
              + (note + "\n" if note else "") + '"""\n'
              'import base64 as _b64\nimport json as _json\n'
              'import zlib as _zlib\n')
    footer = ('\n\n_BP = _json.loads(_zlib.decompress(_b64.b85decode(\n'
              f'    "{blob}"\n)))\n'
              '_EXEC = ReferenceExecutor(_BP)\n\n\n'
              'def agent(obs):\n'
              '    try:\n'
              '        return _EXEC.act(obs)\n'
              '    except Exception:\n'
              '        return {"farmer": ["PASS"], "hands": [],'
              ' "market": []}\n')
    src = header + "\n" + facts + "\n\n" + execu + footer
    path = out / "main.py"
    path.write_text(src)
    return path


def _load_agent(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission_check",
                                                  str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def _episode(agent_fn, seat, seed, merge_obs):
    """Play one episode with the candidate at `seat` vs idle. Returns
    (bank, worst_turn_seconds, n_pass_turns)."""
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": int(seed)})
    env.reset(2)
    worst, n_pass = 0.0, 0
    while not env.done:
        if merge_obs:
            shared = env.state[0].observation
            obs = dict(env.state[seat].observation)
            for k, v in shared.items():
                if obs.get(k) is None and k != "player":
                    obs[k] = v
        else:
            obs = env.state[seat].observation
        t0 = time.time()
        act = agent_fn(obs)
        worst = max(worst, time.time() - t0)
        if act == PASS_ACT or (act.get("farmer") == ["PASS"]
                               and not act.get("market")):
            n_pass += 1
        acts = [None, None]
        acts[seat] = act
        acts[1 - seat] = dict(PASS_ACT)
        env.step(acts)
    return float(env.state[seat].reward or 0), worst, n_pass


def precheck(path, seed=11, floor=5000):
    src = Path(path).read_text()
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:12s} {detail}")

    print(f"prechecking {path}")
    try:
        compile(src, str(path), "exec")
        check("syntax", True, f"{len(src):,} bytes")
    except SyntaxError as e:
        check("syntax", False, str(e))
        return False

    imports = {n.name.split(".")[0]
               for node in ast.walk(ast.parse(src))
               if isinstance(node, ast.Import) for n in node.names}
    imports |= {node.module.split(".")[0]
                for node in ast.walk(ast.parse(src))
                if isinstance(node, ast.ImportFrom) and node.module}
    bad = imports - ALLOWED_IMPORTS
    check("stdlib-only", not bad, f"imports {sorted(imports)}"
          + (f" NON-STDLIB {sorted(bad)}" if bad else ""))

    agent_fn = _load_agent(path)
    bank0, worst0, _ = _episode(agent_fn, 0, seed, merge_obs=True)
    check("smoke", bank0 >= floor,
          f"seat 0 bank ${bank0:,.0f} (floor ${floor:,})")

    agent_fn = _load_agent(path)
    bank1, worst1, n_pass1 = _episode(agent_fn, 1, seed, merge_obs=False)
    check("seat 1 raw", bank1 >= floor and n_pass1 < 360,
          f"seat 1 bank ${bank1:,.0f}, {n_pass1} pass-turns "
          "(trimmed observation)")

    worst = max(worst0, worst1)
    check("latency", worst < 0.9, f"worst turn {worst*1000:.0f} ms "
          "(budget 1000 ms)")

    agent_fn = _load_agent(path)
    bank0b, _, _ = _episode(agent_fn, 0, seed, merge_obs=True)
    check("determinism", abs(bank0b - bank0) < 0.5,
          f"replayed bank ${bank0b:,.0f} vs ${bank0:,.0f}")

    ok = all(c[1] for c in checks)
    print(f"precheck: {'ALL PASS' if ok else 'FAILED'} "
          f"({sum(1 for c in checks if c[1])}/{len(checks)})")
    if ok:
        print(f"submit with:\n  kaggle competitions submit "
              f"kaggriculture -f {path} -m \"island GA schedule\"")
    return ok
