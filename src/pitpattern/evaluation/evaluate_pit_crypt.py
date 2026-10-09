#!/usr/bin/env python3
"""Seed statistics for the crypt lobulation model (labels from simulate_pit_crypt.label).

For every p: label counts over seeds, Wilson 95% interval of the target label,
width ratio of the rendered pattern to the Type I width (median and p90),
forks in the rendered image, and how many units changed during the
low-temperature holds (long-time stability).

    uv run python -m pitpattern.evaluation.evaluate_pit_crypt 'artifacts/pit_crypt/validation/seed*/metrics.csv'
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pitpattern.evaluation.evaluate_pit_groove import wilson

TARGET = {"I": (0.0, 0.5), "IIIL": (0.5, 1.3), "IV_B": (1.9, 2.7), "IV_V": (2.7, 99)}
ORDER = ("I", "IIIL", "IV_B", "IV_V", "transition")


def target_of(p: float) -> str:
    for k, (lo, hi) in TARGET.items():
        if lo <= p < hi:
            return k
    return "transition"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rows = [r for pat in a.metrics for f in sorted(glob.glob(pat)) for r in csv.DictReader(open(f))]
    by_p = defaultdict(list)
    for r in rows:
        by_p[float(r["p"])].append(r)
    # w0 = opening diameter of the Type I pits (area-equivalent; the skeleton estimate is biased for disks)
    w0 = np.median([float(r["equiv_diameter_median"]) for r in rows if r["label"] == "I"])
    holds = sorted({k for r in rows for k in r if k.startswith("units_changed_after_")},
                   key=lambda k: int(k.rsplit("_", 1)[1]))
    out = []
    print(f"w0 (Type I pit opening diameter) = {w0:.2f} px")
    print("    p   n  " + " ".join(f"{k:>5s}" for k in ORDER) + "  target   ok   95%CI       w/w0  p90/w0  forks  "
          + " ".join(h.replace("units_changed_after_", "chg@") for h in holds))
    for p in sorted(by_p):
        rs = by_p[p]
        n = len(rs)
        cnt = {k: sum(r["label"] == k for r in rs) for k in ORDER}
        tgt = target_of(p)
        k_ok = cnt.get(tgt, 0)
        lo, hi = wilson(k_ok, n)
        wm = np.median([float(r["feature_width_median"]) for r in rs]) / w0
        wp = np.max([float(r["feature_width_p90"]) for r in rs]) / w0
        forks = np.median([float(r["fork_count"]) for r in rs])
        chg = [max(int(r[h]) for r in rs) for h in holds]
        print(f"{p:5.2f} {n:3d}  " + " ".join(f"{cnt[k]:5d}" for k in ORDER)
              + f"  {tgt:>6s} {k_ok:2d}/{n} [{lo:.2f},{hi:.2f}]  {wm:5.2f}  {wp:5.2f}  {forks:5.0f}  "
              + " ".join(f"{c:5d}" for c in chg))
        out.append(dict(p=p, n=n, counts=cnt, target=tgt, target_ok=k_ok, wilson95=[lo, hi], width_ratio_median=wm,
                        width_ratio_p90_max=wp, forks_median=forks, max_units_changed={h: c for h, c in zip(holds, chg)}))
    if a.json:
        a.json.write_text(json.dumps(dict(w0=w0, points=out), indent=2))


if __name__ == "__main__":
    main()
