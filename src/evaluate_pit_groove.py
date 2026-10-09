#!/usr/bin/env python3
"""Goal labels and seed statistics for the crest-groove model.

The labels follow docs/pit_pattern_goal.md section 6.2 and were fixed on the
exploration seeds (1, 7, 42) before the validation seeds (100-119) were run.
They use the project goal metrics plus the staircase-robust fork counts of
pit_fork_metrics; lengths are in units of the crest spacing lambda.

  I     no fork, p90 component elongation <= 2, p90 skeleton length <= lambda/4
  IIIL  at most 2 forks, elongation > 2, p90 skeleton length <= lambda, no
        unit dominates (longest fraction < 0.1)       -> short tubes
  IV_B  >= 4 forks spread over >= 2 units, longest fraction <= 0.40
                                                       -> several finite trees
  IV_V  longest fraction >= 0.5, >= 15 enclosed crest islands, >= 20 forks
                                                       -> connected groove mesh
  other everything else (transition / unclassified, reported, never dropped)

Width: median / w0 <= 1.5 for every label and p90 / w0 <= 2.0 for IV_B/IV_V,
where w0 is the Type I median width on the exploration seeds (one value for
all validation runs).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

LAMBDA = 24.0
ORDER = ("I", "IIIL", "IV_B", "IV_V")


def label(row: dict, wavelength: float = LAMBDA) -> str:
    g = lambda k: float(row[k])
    if g("coverage") < 0.01:
        return "empty"
    forks, fcomp = g("fork_count"), g("fork_components")
    elong = g("p90_component_elongation")
    ulen90 = g("unit_length_p90")
    lcf = g("longest_component_fraction")
    islands = g("background_island_count")
    if forks == 0 and elong <= 2.0 and ulen90 <= wavelength / 4:
        return "I"
    if forks <= 2 and elong > 2.0 and ulen90 <= wavelength and lcf < 0.1:
        return "IIIL"
    if forks >= 4 and fcomp >= 2 and lcf <= 0.40:
        return "IV_B"
    if lcf >= 0.5 and islands >= 15 and forks >= 20:
        return "IV_V"
    return "other"


def width_ok(row: dict, lab: str, w0: float) -> bool:
    med = float(row["feature_width_median"]) / w0
    p90 = float(row["feature_width_p90"]) / w0
    if med > 1.5:
        return False
    if lab in ("IV_B", "IV_V") and p90 > 2.0:
        return False
    return True


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            rows.extend(csv.DictReader(open(f)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", nargs="+", help="metrics.csv paths or globs")
    p.add_argument("--w0", type=float, required=True, help="Type I reference width from exploration")
    p.add_argument("--checkpoint", default="1T")
    p.add_argument("--wavelength", type=float, default=LAMBDA)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    rows = [r for r in load(a.metrics) if r["checkpoint"] == a.checkpoint]
    table: dict[float, list[tuple[int, str, bool]]] = defaultdict(list)
    for r in rows:
        lab = label(r, a.wavelength)
        table[float(r["s"])].append((int(r["seed"]), lab, width_ok(r, lab, a.w0)))
    out = []
    print(f"checkpoint {a.checkpoint}  w0={a.w0}")
    print("     s   n  " + "  ".join(f"{k:>5s}" for k in (*ORDER, "other", "empty")) + "  top(width-ok)  95%CI")
    for s in sorted(table):
        entries = table[s]
        n = len(entries)
        counts = {k: sum(1 for _, l, _ in entries if l == k) for k in (*ORDER, "other", "empty")}
        top = max(ORDER, key=lambda k: counts[k])
        k_ok = sum(1 for _, l, ok in entries if l == top and ok)
        lo, hi = wilson(k_ok, n)
        print(f"{s:+.4f} {n:3d}  " + "  ".join(f"{counts[k]:5d}" for k in (*ORDER, "other", "empty"))
              + f"  {top:>5s} {k_ok:2d}/{n}  [{lo:.2f},{hi:.2f}]")
        out.append(dict(s=s, n=n, counts=counts, top=top, top_width_ok=k_ok, wilson95=[lo, hi],
                        per_seed=[dict(seed=sd, label=l, width_ok=ok) for sd, l, ok in sorted(entries)]))
    if a.json:
        a.json.write_text(json.dumps(dict(checkpoint=a.checkpoint, w0=a.w0, points=out), indent=2))


if __name__ == "__main__":
    main()
