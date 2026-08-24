"""The top-N pool: rotating snapshots of the best genomes found.

One JSON file, shareable across runs: successive searches feed the same
pool, members rotate out as better genomes arrive, and the arena and
submit commands read from it. Membership is decided on the SCREEN mean;
the arena is where members earn a head-to-head ranking.
"""
import json
import time
from pathlib import Path

from .genome import gid_of


class Pool:
    def __init__(self, path, size=10):
        self.path = Path(path)
        self.size = int(size)
        self.members = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.members = data.get("members", [])
            self.size = int(data.get("size", self.size))

    def offer(self, genome, screen_banks, origin=""):
        """Insert if it beats the pool's worst (or the pool has room).
        Returns True when the pool changed."""
        gid = gid_of(genome)
        mean = sum(screen_banks.values()) / max(1, len(screen_banks))
        for m in self.members:
            if m["gid"] == gid:
                return False
        entry = {"gid": gid, "screen_mean": round(mean, 1),
                 "screen_banks": {str(k): v
                                  for k, v in screen_banks.items()},
                 "genome": genome, "origin": origin,
                 "added": int(time.time())}
        self.members.append(entry)
        self.members.sort(key=lambda m: -m["screen_mean"])
        self.members = self.members[:self.size]
        if any(m["gid"] == gid for m in self.members):
            self.save()
            return True
        return False

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"size": self.size, "members": self.members}, indent=1))

    def table(self):
        rows = [f"{'#':>2s} {'gid':10s} {'screen':>9s} {'origin':18s}"]
        for i, m in enumerate(self.members, 1):
            rows.append(f"{i:2d} {m['gid']:10s} "
                        f"{m['screen_mean']:9,.0f} {m['origin'][:18]:18s}")
        return "\n".join(rows)
