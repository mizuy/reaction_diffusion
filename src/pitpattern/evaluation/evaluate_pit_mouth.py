#!/usr/bin/env python3
"""Image-based stage label and seed statistics for the crypt-mouth model.

`measure` works on the rendered binary pattern only (purple = True, periodic box):
  units      every purple component is a dot / tube / Y / other from its centreline
             (length and number of free ends)
  grooves    share of the centreline that lies in long grooves
  islands    share of the red area that lies in enclosed islands
`label` turns the measures into Round / Elongated / Branch / Gyrus / Villous / transition.
The thresholds were fixed on the exploration seeds before the validation seeds were run.

    uv run python -m pitpattern.evaluation.evaluate_pit_mouth 'artifacts/pit_mouth/validation/seed*/metrics.csv'
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitpattern.evaluation.evaluate_pit_groove import wilson
from pitpattern.metrics.pit_fork_metrics import crossing_number

ORDER = ("Round", "Elongated", "Branch", "Gyrus", "Villous", "transition")
TARGET = {"Round": (0.0, 0.45), "Elongated": (0.55, 1.45), "Branch": (1.5, 2.55), "Gyrus": (2.85, 3.25), "Villous": (3.45, 4.01)}


def periodic_label(mask: np.ndarray, connectivity: int = 8):
    """Connected components of a periodic image -> (labels 0..n, areas)."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    shifts = (-1, 0, 1) if connectivity == 8 else (0,)
    for a, b in ((lab[0], lab[-1]), (lab[:, 0], lab[:, -1])):
        for s in shifts:
            bb = np.roll(b, s)
            for i, j in set(zip(a[(a > 0) & (bb > 0)].tolist(), bb[(a > 0) & (bb > 0)].tolist())):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    root = np.array([find(i) for i in range(n)])
    uniq, inv = np.unique(root, return_inverse=True)       # background root 0 stays label 0
    lab = inv[lab]
    return lab, np.bincount(lab.ravel(), minlength=len(uniq))


def window_label(mask: np.ndarray, connectivity: int = 8):
    """Components of a non-periodic window -> (labels, areas, touches the border)."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    edge = np.zeros(n, bool)
    for line in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        edge[np.unique(line)] = True
    return lab, np.bincount(lab.ravel(), minlength=n), edge


def measure(binary: np.ndarray, cfg, periodic: bool = True) -> dict[str, float]:
    """periodic=False: a window cut out of a larger pattern; units cut by the border are not counted."""
    from pitpattern.models.simulate_pit_mouth import centreline

    D = cfg.D
    binary = binary.astype(bool)
    out: dict[str, float] = dict(purple_fraction=float(binary.mean()))
    if not binary.any() or binary.all():
        return out | dict(n_components=0, frac_dot=0.0, frac_tube=0.0, frac_y=0.0, elongation_median=0.0,
                          long_groove_fraction=0.0, red_island_fraction=0.0, n_red_islands=0,
                          width_median=0.0, width_p90=0.0, gap_median=0.0, void_max=0.0, largest_purple_fraction=0.0)
    mode = "wrap" if periodic else "edge"
    sk = centreline(binary, periodic=periodic)
    if periodic:
        lab, area = periodic_label(binary, 8)
        whole = np.ones(len(area) - 1, bool)
    else:
        lab, area, edge = window_label(binary, 8)
        whole = ~edge[1:]
    n = len(area)
    cn = crossing_number(np.pad(sk, 2, mode="wrap" if periodic else "constant"))[2:-2, 2:-2]     # 1 at a free end
    ends = np.bincount(lab[sk & (cn == 1)], minlength=n)[1:]
    length = np.bincount(lab[sk], minlength=n)[1:].astype(float)
    pad = 40
    dist = cv2.distanceTransform(np.pad(binary, pad, mode=mode).astype(np.uint8), cv2.DIST_L2, 5)[pad:-pad, pad:-pad]
    w = 2 * dist[sk]
    width = float(np.median(w))
    elong = (length + width) / width
    dot = elong < 1.5
    tube = ~dot & (ends == 2) & (length < 3 * D)
    y = (ends == 3) & (length > 1.2 * width) & (length < 5 * D)
    long_ = length > 6 * D                    # a long groove counts even when the window cuts it
    dot, tube, y = dot & whole, tube & whole, y & whole
    k = max(1, int((whole | long_).sum()))
    out.update(n_components=int((whole | long_).sum()), frac_dot=float(dot.sum() / k), frac_tube=float(tube.sum() / k),
               frac_y=float(y.sum() / k), elongation_median=float(np.median(elong[whole])) if whole.any() else 0.0,
               long_groove_fraction=float(length[length > 6 * D].sum() / max(length.sum(), 1)),
               width_median=width, width_p90=float(np.percentile(w, 90)),
               largest_purple_fraction=float(area[1:].max() / area[1:].sum()))
    # red: enclosed islands and gaps
    if periodic:
        rlab, rarea = periodic_label(~binary, 4)
        isl = rarea[1:] < 12 * D * D
    else:
        rlab, rarea, redge = window_label(~binary, 4)
        isl = (rarea[1:] < 12 * D * D) & ~redge[1:]
        if (~redge[1:]).any():                      # islands cut by the border are left out of both sides
            rarea = rarea * np.concatenate([[True], ~redge[1:] | (rarea[1:] >= 12 * D * D)])
    out.update(red_island_fraction=float(rarea[1:][isl].sum() / rarea[1:].sum()), n_red_islands=int(isl.sum()))
    rdist = cv2.distanceTransform(np.pad(~binary, pad, mode=mode).astype(np.uint8), cv2.DIST_L2, 5)[pad:-pad, pad:-pad]
    rsk = centreline(~binary, periodic=periodic)
    out.update(gap_median=float(2 * np.median(rdist[rsk])) if rsk.any() else 0.0, void_max=float(2 * rdist.max()))
    return out


def label(m: dict[str, float]) -> str:
    if m["n_components"] == 0:
        return "transition"
    if m["frac_dot"] >= 0.9:
        return "Round"
    if m["frac_tube"] >= 0.9:
        return "Elongated"
    if m["frac_y"] >= 0.75 and m["frac_y"] + m["frac_tube"] >= 0.9:
        return "Branch"
    if m["red_island_fraction"] >= 0.7:
        return "Villous"
    if m["long_groove_fraction"] >= 0.7 and m["red_island_fraction"] <= 0.35:
        return "Gyrus"
    return "transition"


def target_of(p: float) -> str:
    for k, (lo, hi) in TARGET.items():
        if lo <= p <= hi:
            return k
    return "transition"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rows = [r for pat in a.metrics for f in sorted(glob.glob(pat)) for r in csv.DictReader(open(f))]
    holds = sorted({float(r["hold"]) for r in rows})
    base = [r for r in rows if float(r["hold"]) == holds[0]]
    by_p = defaultdict(list)
    for r in base:
        by_p[float(r["p"])].append(r)
    w0 = float(np.median([float(r["width_median"]) for r in base if r["label"] == "Round"]))
    gap0 = float(np.median([float(r["gap_median"]) for r in base if r["label"] == "Round"]))
    last = {(float(r["p"]), r["seed"]): r for r in rows if float(r["hold"]) == holds[-1]}
    out = []
    print(f"w0 (Round pit diameter) = {w0:.2f} px, Round gap = {gap0:.2f} px, holds = {holds}")
    print("    p   n  " + " ".join(f"{k[:6]:>6s}" for k in ORDER) + "     target   ok   95%CI      w/w0 p90/w0  gap/gap0  "
          f"label kept@x{holds[-1]:g}  changed px")
    for p in sorted(by_p):
        rs = by_p[p]
        n = len(rs)
        cnt = {k: sum(r["label"] == k for r in rs) for k in ORDER}
        tgt = target_of(p)
        k_ok = cnt.get(tgt, 0)
        lo, hi = wilson(k_ok, n)
        wm = float(np.median([float(r["width_median"]) for r in rs]) / w0)
        wp = float(np.max([float(r["width_p90"]) for r in rs]) / w0)
        gp = float(np.median([float(r["gap_median"]) for r in rs]) / gap0)
        kept = sum(last[(p, r["seed"])]["label"] == r["label"] for r in rs)
        chg = float(np.max([float(last[(p, r["seed"])]["changed_fraction"]) for r in rs]))
        print(f"{p:5.2f} {n:3d}  " + " ".join(f"{cnt[k]:6d}" for k in ORDER)
              + f"  {tgt:>10s} {k_ok:2d}/{n} [{lo:.2f},{hi:.2f}]  {wm:5.2f}  {wp:5.2f}   {gp:5.2f}      {kept:2d}/{n}        {chg:.3f}")
        out.append(dict(p=p, n=n, counts=cnt, target=tgt, target_ok=k_ok, wilson95=[lo, hi], width_ratio_median=wm,
                        width_ratio_p90_max=wp, gap_ratio_median=gp, label_kept=kept, changed_fraction_max=chg))
    if a.json:
        a.json.write_text(json.dumps(dict(w0=w0, gap0=gap0, holds=holds, points=out), indent=2))


if __name__ == "__main__":
    main()
