#!/usr/bin/env python3
"""Multi-nucleus Laplacian growth (DBM) prototype for Kudo pit patterns.

Model (one scalar field, quasi-static, periodic 2D lattice):

    stroma:   0 = ∇²φ − (φ − 1) / ℓ²        (nutrient relaxes to 1 within a screening length ℓ)
    pit set:  φ = 0                            (Dirichlet on the growing aggregate)
    growth:   perimeter site i is added with probability ∝ φ_i^η · ρ_i^β

φ is solved with a 9-point (isotropic) stencil and 4-colour SOR.  ρ_i is the local pit
fraction in a disc of radius beta_r around the site: β > 0 favours concave sites and
suppresses one-pixel bumps, a cheap stand-in for surface tension (Gibbs-Thomson).
Every nucleus (a Type I pit) is a disc of radius r0.  Each accepted site is stamped as a
disc of radius r_w so that branches have a finite width.  Growth is driven per nucleus
(each pit adds one stamp per "cell division"); the field only decides *where* on the pit
perimeter the stamp goes.  The single sweep parameter s ∈ [0, 1] is the normalised growth
budget (mass per nucleus); η may ramp with s (eta -> eta_end over s ∈ [0, eta_ramp]).

Usage examples:

    python3 src/laplacian_pit.py grid   --out /tmp/lap --size 192 --spacing 48 --etas 0,1,2,3,4,6 \
        --r-w 1 --beta 4 --beta-r 2 --mass-max 1200 --zoom 128
    python3 src/laplacian_pit.py phases --out /tmp/lap --size 256 --spacing 48 --r0 2.5 \
        --eta 0 --eta-end 4 --eta-ramp 0.15 --r-w 1 --beta 4 --beta-r 2 --mass-max 1400 \
        --s-phases 0.02,0.08,0.4,0.85
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pit_steps import (  # noqa: E402
    prepare_skeleton_topology,
    skeleton_endpoints,
    zhang_suen_thin,
)


# --------------------------------------------------------------------------- model


@dataclass
class LapParams:
    size: int = 192
    spacing: float = 48.0          # nucleus spacing in px (jittered hex lattice)
    jitter: float = 0.35           # jitter as fraction of spacing
    r0: float = 2.0                # nucleus (Type I pit) radius
    r_w: float = 1.0               # stamp radius (branch half-width)
    ell: float = 32.0              # screening length of the nutrient field
    eta: float = 2.0               # DBM exponent
    eta_end: float | None = None   # if set, eta ramps linearly eta -> eta_end over s in [0, eta_ramp]
    eta_ramp: float = 1.0
    r_w_end: float | None = None   # if set, stamp radius ramps r_w -> r_w_end over s
    mass_max: float = 1200.0       # per-nucleus mass (px) at s = 1
    noise_m: int = 1               # noise reduction: a site must be picked m times before it is added
    beta: float = 4.0              # surface tension proxy: weight ∝ (local pit fraction)^beta
    beta_r: float = 2.0            # radius of the neighbourhood used for the local pit fraction
    smooth_sigma: float = 1.0      # rendering/metric resolution (px); 0 = raw lattice
    growth: str = "per_nucleus"    # or "global"
    grow_frac: float = 0.25        # fraction of nuclei that grow one stamp per field solve
    sor_omega: float = 1.8
    sor_iters: int = 24
    sor_init_iters: int = 400
    seed: int = 7
    lattice: str = "hex"           # or "random"


def disc_offsets(r: float) -> np.ndarray:
    n = int(math.ceil(r))
    ys, xs = np.mgrid[-n : n + 1, -n : n + 1]
    keep = ys * ys + xs * xs <= r * r + 1e-9
    return np.stack([ys[keep], xs[keep]], axis=1)


def hex_nuclei(size: int, spacing: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    dy = spacing * math.sqrt(3) / 2
    ny = max(1, int(round(size / dy)))
    nx = max(1, int(round(size / spacing)))
    dy_eff = size / ny
    dx_eff = size / nx
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5 * (j % 2)) * dx_eff
            y = (j + 0.5) * dy_eff
            pts.append((y, x))
    pts = np.array(pts, dtype=np.float64)
    pts += rng.uniform(-jitter * spacing, jitter * spacing, size=pts.shape)
    return np.mod(pts, size)


def random_nuclei(size: int, spacing: float, rng: np.random.Generator) -> np.ndarray:
    n = int(round(size * size / (spacing * spacing * math.sqrt(3) / 2)))
    pts = []
    tries = 0
    min_d = 0.55 * spacing
    while len(pts) < n and tries < 20000:
        tries += 1
        p = rng.uniform(0, size, size=2)
        ok = True
        for q in pts:
            d = np.abs(p - q)
            d = np.minimum(d, size - d)
            if d @ d < min_d * min_d:
                ok = False
                break
        if ok:
            pts.append(p)
    return np.array(pts, dtype=np.float64)


class LaplacianPits:
    """Multi-nucleus screened-Laplacian (DBM) growth on a periodic lattice."""

    def __init__(self, p: LapParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        n = p.size
        self.label = np.zeros((n, n), dtype=np.int32)
        self.phi = np.ones((n, n), dtype=np.float64)
        if p.lattice == "random":
            self.nuclei = random_nuclei(n, p.spacing, self.rng)
        else:
            self.nuclei = hex_nuclei(n, p.spacing, p.jitter, self.rng)
        self.n_nuc = len(self.nuclei)
        self.mass = np.zeros(self.n_nuc + 1, dtype=np.int64)
        self.votes = np.zeros((n, n), dtype=np.int16)
        for j, (y, x) in enumerate(self.nuclei, start=1):
            self._stamp(int(round(y)) % n, int(round(x)) % n, j, p.r0)
        self.mass0 = self.mass.copy()
        yy, xx = np.mgrid[0:n, 0:n]
        self.colors = [((yy % 2) == a) & ((xx % 2) == b) for a in (0, 1) for b in (0, 1)]
        self.s = 0.0
        self.solve(p.sor_init_iters)

    # -- geometry -----------------------------------------------------------
    def _stamp(self, y: int, x: int, j: int, r: float) -> int:
        n = self.p.size
        off = disc_offsets(r)
        ys = (y + off[:, 0]) % n
        xs = (x + off[:, 1]) % n
        free = self.label[ys, xs] == 0
        self.label[ys[free], xs[free]] = j
        self.phi[ys[free], xs[free]] = 0.0
        self.mass[j] += int(free.sum())
        return int(free.sum())

    def _vote(self, y: int, x: int, j: int, r: float) -> int:
        """Noise reduction: the site is added only after noise_m selections."""
        self.votes[y, x] += 1
        if self.votes[y, x] < self.p.noise_m:
            return 0
        self._stamp(y, x, j, r)
        return 1

    @property
    def agg(self) -> np.ndarray:
        return self.label > 0

    def coverage(self) -> float:
        return float(self.agg.mean())

    # -- field -------------------------------------------------------------
    @staticmethod
    def _nb9(phi: np.ndarray) -> np.ndarray:
        """Weighted neighbour sum of the isotropic 9-point Laplacian (edges 4, corners 1)."""
        up, dn = np.roll(phi, 1, 0), np.roll(phi, -1, 0)
        e = up + dn + np.roll(phi, 1, 1) + np.roll(phi, -1, 1)
        c = np.roll(up, 1, 1) + np.roll(up, -1, 1) + np.roll(dn, 1, 1) + np.roll(dn, -1, 1)
        return 4.0 * e + c

    def solve(self, iters: int) -> float:
        """Jacobi/SOR-type relaxation of (∇² − 1/ℓ²)φ = −1/ℓ² on free cells, φ=0 on the pit set.

        9-point stencil: ∇²φ ≈ (4Σedge + Σcorner − 20φ)/6, which is isotropic to 4th order
        and greatly reduces the square-lattice anisotropy of DBM.
        """
        k2 = 1.0 / (self.p.ell * self.p.ell)
        w = self.p.sor_omega
        agg = self.agg
        phi = self.phi
        phi[agg] = 0.0
        denom = 20.0 + 6.0 * k2
        src = 6.0 * k2
        free = ~agg
        for _ in range(iters):
            # 4-colour Gauss-Seidel: with the 9-point stencil every neighbour of a
            # cell has a different colour, so over-relaxation is stable.
            for mask in self.colors:
                new = (self._nb9(phi) + src) / denom
                upd = mask & free
                phi[upd] = (1 - w) * phi[upd] + w * new[upd]
        res = np.abs(self._nb9(phi) - denom * phi + src) / 6.0
        res[agg] = 0.0
        return float(res.max())

    # -- growth ------------------------------------------------------------
    def perimeter(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Free cells 8-adjacent to the aggregate, with the label of one neighbour."""
        lab = self.label
        agg = lab > 0
        cand = np.zeros_like(lab)
        shifts = [(1, 0), (-1, 0), (1, 1), (-1, 1)]
        for sh, ax in shifts:
            nl = np.roll(lab, sh, ax)
            take = (cand == 0) & (nl > 0)
            cand[take] = nl[take]
        for sy in (1, -1):
            for sx in (1, -1):
                nl = np.roll(np.roll(lab, sy, 0), sx, 1)
                take = (cand == 0) & (nl > 0)
                cand[take] = nl[take]
        per = (~agg) & (cand > 0)
        ys, xs = np.nonzero(per)
        return ys, xs, cand[ys, xs]

    def _local_fraction(self) -> np.ndarray:
        """Fraction of pit cells in a disc of radius beta_r around each cell (periodic)."""
        off = disc_offsets(self.p.beta_r)
        k = np.zeros((off[:, 0].max() * 2 + 1,) * 2, dtype=np.float32)
        k[off[:, 0] + off[:, 0].max(), off[:, 1] + off[:, 1].max()] = 1.0 / len(off)
        pad = off[:, 0].max()
        a = np.pad(self.agg.astype(np.float32), pad, mode="wrap")
        return cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REPLICATE)[pad:-pad, pad:-pad]

    def eta_now(self) -> float:
        p = self.p
        if p.eta_end is None:
            return p.eta
        return p.eta + (p.eta_end - p.eta) * min(1.0, self.s / max(p.eta_ramp, 1e-9))

    def r_w_now(self) -> float:
        p = self.p
        if p.r_w_end is None:
            return p.r_w
        return p.r_w + (p.r_w_end - p.r_w) * min(1.0, self.s)

    def step(self, s_target: float | None = None) -> int:
        """One field solve + a batch of stamps. Returns number of stamps placed."""
        p = self.p
        ys, xs, labs = self.perimeter()
        if ys.size == 0:
            return 0
        phi = np.maximum(self.phi[ys, xs], 1e-300)
        eta = self.eta_now()
        r_w = self.r_w_now()
        # Concave perimeter sites (many pit neighbours) are favoured, convex bumps are
        # suppressed: a cheap stand-in for the Gibbs-Thomson curvature correction.
        tension = self._local_fraction()[ys, xs] ** p.beta if p.beta > 0 else np.ones_like(phi)
        if s_target is None:
            s_target = 1.0
        target = p.mass_max * min(1.0, s_target) + self.mass0[1:]
        placed = 0
        if p.growth == "per_nucleus":
            n_grow = max(1, int(round(p.grow_frac * self.n_nuc)))
            order = self.rng.permutation(self.n_nuc) + 1
            for j in order[:n_grow]:
                if self.mass[j] >= target[j - 1]:
                    continue
                sel = np.nonzero(labs == j)[0]
                if sel.size == 0:
                    continue
                w = phi[sel]
                w = (w / w.max()) ** eta * tension[sel]
                w = w / w.sum()
                i = sel[self.rng.choice(sel.size, p=w)]
                placed += self._vote(int(ys[i]), int(xs[i]), int(j), r_w)
        else:
            n_grow = max(1, int(round(p.grow_frac * self.n_nuc)))
            open_mask = self.mass[labs] < target[labs - 1]
            sel = np.nonzero(open_mask)[0]
            if sel.size == 0:
                return 0
            w = phi[sel]
            w = (w / w.max()) ** eta * tension[sel]
            w = w / w.sum()
            k = min(n_grow, int((w > 0).sum()))
            picks = self.rng.choice(sel.size, size=k, replace=False, p=w)
            for i in sel[picks]:
                if self.label[ys[i], xs[i]] == 0:
                    placed += self._vote(int(ys[i]), int(xs[i]), int(labs[i]), r_w)
        self.solve(p.sor_iters)
        return placed

    def run_to(self, s_target: float, callback=None, max_steps: int = 200000) -> None:
        """Advance the sweep parameter s to s_target, growing until the mass budget is met."""
        p = self.p
        stall = 0
        for _ in range(max_steps):
            need = p.mass_max * min(1.0, s_target) + self.mass0[1:] - self.mass[1:]
            if need.max() <= 0:
                break
            # s tracks the actual mean mass so that eta(s), r_w(s) follow the growth
            self.s = float(np.clip((self.mass[1:] - self.mass0[1:]).mean() / p.mass_max, 0, s_target))
            placed = self.step(s_target)
            if callback is not None:
                callback(self)
            stall = stall + 1 if placed == 0 else 0
            if stall > 40 * p.noise_m:
                # remaining nuclei are fully enclosed; give up on them
                break
        self.s = s_target


# --------------------------------------------------------------------------- metrics


def fill_small_holes(fg: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Fill enclosed background pockets below the pit-width resolution."""
    fg = fg.astype(bool).copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats((~fg).astype(np.uint8), 4)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            fg[lab == i] = True
    return fg


def smooth_figure(fg: np.ndarray, sigma: float = 1.0, min_area: int = 6) -> np.ndarray:
    """Pit set as seen at the resolution of the stain: Gaussian blur (periodic) + 0.5 threshold,
    then small pockets filled.  This mirrors the Otsu binarisation of the smooth GS field."""
    if sigma <= 0:
        return fill_small_holes(fg, min_area)
    f = fg.astype(np.float32)
    pad = int(3 * sigma) + 1
    fp = np.pad(f, pad, mode="wrap")
    fb = cv2.GaussianBlur(fp, (0, 0), sigma)[pad:-pad, pad:-pad]
    return fill_small_holes(fb >= 0.5, min_area)


def skeleton_graph(sk: np.ndarray, min_arm: int = 4, merge_len: int = 3) -> dict:
    """Junction graph of a skeleton with nearby junction clusters merged.

    Zhang-Suen skeletons of 3-4 px wide branches split one anatomical fork into two
    degree-3 pixels joined by a 1-2 px corridor; the strict repo counter then sees arms of
    length 1-2 and rejects the fork.  Here junction clusters connected by corridors shorter
    than ``merge_len`` are merged, then a node with exactly 3 arms (each >= ``min_arm``) is a
    Y fork and a node with 4 such arms an X.  Returns counts and a merged-node label image.
    """
    sk = sk.astype(bool)
    deg = np.zeros_like(sk, dtype=np.int16)
    if sk.any():
        k = np.ones((3, 3), dtype=np.uint8)
        pad = np.pad(sk.astype(np.uint8), 1, mode="wrap")
        deg = (cv2.filter2D(pad, -1, k, borderType=cv2.BORDER_REPLICATE)[1:-1, 1:-1] - 1).astype(np.int16)
    junc = sk & (deg >= 3)
    nj, jl = cv2.connectedComponents(junc.astype(np.uint8), connectivity=8)
    corr = sk & (deg <= 2)
    nc, cl = cv2.connectedComponents(corr.astype(np.uint8), connectivity=8)
    parent = list(range(nj))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # adjacency corridor -> junction clusters (8-neighbourhood, periodic)
    k3 = np.ones((3, 3), dtype=np.uint8)
    jl_pad = np.pad(jl.astype(np.int32), 1, mode="wrap")
    adj: list[set[int]] = [set() for _ in range(nc)]
    lengths = np.bincount(cl.ravel(), minlength=nc)
    ys, xs = np.nonzero(corr)
    for y, x in zip(ys, xs):
        win = jl_pad[y : y + 3, x : x + 3]
        for v in np.unique(win):
            if v > 0:
                adj[cl[y, x]].add(int(v))
    for c in range(1, nc):
        if lengths[c] < merge_len and len(adj[c]) >= 2:
            a = list(adj[c])
            for b in a[1:]:
                ra, rb = find(a[0]), find(b)
                if ra != rb:
                    parent[rb] = ra
    node_of = [find(j) for j in range(nj)]
    arms: dict[int, list[int]] = {}
    for c in range(1, nc):
        nodes = [node_of[j] for j in adj[c]]
        if not nodes:
            continue
        if len(set(nodes)) == 1 and len(nodes) >= 2 and lengths[c] < merge_len:
            continue  # internal corridor of a merged node
        for nd in nodes:
            arms.setdefault(nd, []).append(int(lengths[c]))
    y3 = x4 = 0
    y3_nodes = []
    for nd, ls in arms.items():
        good = [l for l in ls if l >= min_arm]
        if len(ls) == 3 and len(good) == 3:
            y3 += 1
            y3_nodes.append(nd)
        elif len(ls) == 4 and len(good) >= 3:
            x4 += 1
    merged = np.zeros_like(jl, dtype=np.int32)
    if nj > 1:
        lut = np.array(node_of, dtype=np.int32)
        merged = lut[jl]
    return dict(junc3=y3, junc4=x4, merged_labels=merged, y3_nodes=y3_nodes, n_nodes=len(arms))


def topo_metrics(fg: np.ndarray, roots: np.ndarray | None = None) -> dict:
    """Skeleton topology of a binary figure (same definitions as topo_all.py)."""
    fg = fg.astype(bool)
    sk, br3 = prepare_skeleton_topology(zhang_suen_thin(fg))
    n3, br_lab = cv2.connectedComponents(br3.astype(np.uint8), connectivity=8)
    n_sk, sk_lab, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), 8)
    nb, _, _, _ = cv2.connectedComponentsWithStats((~fg).astype(np.uint8), 4)
    nf, fg_lab, _, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    ep = skeleton_endpoints(sk)
    g = skeleton_graph(sk, min_arm=3, merge_len=3)
    out = dict(
        cov=round(float(fg.mean()), 3),
        junc3=int(g["junc3"]),
        junc3_strict=int(n3 - 1),
        junc4=int(g["junc4"]),
        endpoints=int(ep.sum()),
        sk_comps=int(n_sk - 1),
        n_fg=int(nf - 1),
        n_bg=int(nb - 1),
        euler=int(nf - nb),
        sk_px=int(sk.sum()),
    )
    # label image of merged Y nodes (0 elsewhere), used for per-tree counts and generations
    y_lab = np.zeros_like(sk_lab, dtype=np.int32)
    if g["y3_nodes"]:
        lut = np.zeros(int(g["merged_labels"].max()) + 1, dtype=np.int32)
        for i, nd in enumerate(g["y3_nodes"], start=1):
            lut[nd] = i
        y_lab = lut[g["merged_labels"]]
    n_y = len(g["y3_nodes"])
    # per-component junction counts -> trees and generation depth
    junc_per_comp = np.zeros(n_sk, dtype=int)
    for lab in range(1, n_y + 1):
        ys, xs = np.nonzero(y_lab == lab)
        if ys.size:
            junc_per_comp[sk_lab[ys[0], xs[0]]] += 1
    trees = int((junc_per_comp[1:] > 0).sum())
    out["trees"] = trees
    out["junc_per_tree_max"] = int(junc_per_comp[1:].max()) if n_sk > 1 else 0
    out["junc_per_tree_mean"] = round(float(junc_per_comp[1:].mean()), 2) if n_sk > 1 else 0.0
    # generation depth: Y nodes along the longest root->tip skeleton path
    if roots is not None and n_y > 0:
        out["gen_max"], out["gen_mean"] = generation_depth(sk, y_lab, roots)
    else:
        out["gen_max"], out["gen_mean"] = (1 if fg.any() else 0), (1.0 if fg.any() else 0.0)
    # width estimate: distance transform on the figure
    dt = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 3)
    if sk.any():
        w = 2 * dt[sk]
        out["width_mean"] = round(float(w.mean()), 2)
        out["width_cv"] = round(float(w.std() / max(w.mean(), 1e-6)), 2)
    else:
        out["width_mean"] = out["width_cv"] = 0.0
    # island (background component) size stats for the inverted phase
    if nb > 2:
        _, _, bst, _ = cv2.connectedComponentsWithStats((~fg).astype(np.uint8), 4)
        areas = bst[1:, cv2.CC_STAT_AREA]
        out["island_area_med"] = float(np.median(areas))
    else:
        out["island_area_med"] = float("nan")
    return out


def generation_depth(sk: np.ndarray, br_lab: np.ndarray, roots: np.ndarray) -> tuple[int, float]:
    """BFS on the skeleton from the pixel nearest each root; generation = 1 + number of
    distinct junction clusters on the path (max over tips)."""
    n = sk.shape[0]
    sk_ys, sk_xs = np.nonzero(sk)
    if sk_ys.size == 0:
        return 0, 0.0
    pts = np.stack([sk_ys, sk_xs], axis=1)
    gens = []
    visited_global = np.zeros_like(sk, dtype=bool)
    for ry, rx in roots:
        d = np.abs(pts - np.array([ry, rx]))
        d = np.minimum(d, n - d)
        i = int(np.argmin((d * d).sum(1)))
        sy, sx = int(pts[i, 0]), int(pts[i, 1])
        if visited_global[sy, sx] or (d[i] ** 2).sum() > 36:
            continue
        # BFS
        from collections import deque

        q = deque([(sy, sx, 0, -1)])
        seen = {(sy, sx)}
        best = 0
        while q:
            y, x, g, last = q.popleft()
            lab = int(br_lab[y, x])
            if lab > 0 and lab != last:
                g += 1
                last = lab
            best = max(best, g)
            visited_global[y, x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = (y + dy) % n, (x + dx) % n
                    if sk[ny, nx] and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        q.append((ny, nx, g, last))
        gens.append(best + 1)
    if not gens:
        return 0, 0.0
    return int(max(gens)), round(float(np.mean(gens)), 2)


def branch_angles(fg: np.ndarray, R: int = 7) -> list[float]:
    """Opening angles (deg) of Y forks: for each merged 3-arm node, skeleton pixels reachable
    within radius R are traced, the three arm directions are the angular clusters at distance
    ~R, and the smallest of the three inter-arm angles is returned (the daughter opening)."""
    from collections import deque

    sk, _ = prepare_skeleton_topology(zhang_suen_thin(fg.astype(bool)))
    g = skeleton_graph(sk, min_arm=3, merge_len=3)
    n = sk.shape[0]
    angles: list[float] = []
    for nd in g["y3_nodes"]:
        ys, xs = np.nonzero(g["merged_labels"] == nd)
        if ys.size == 0:
            continue
        # unwrap the cluster around its first pixel (periodic domain)
        cy0, cx0 = int(ys[0]), int(xs[0])
        cy = cy0 + np.mean(((ys - cy0 + n // 2) % n) - n // 2)
        cx = cx0 + np.mean(((xs - cx0 + n // 2) % n) - n // 2)
        q = deque((int(y), int(x), 0.0) for y, x in zip(ys, xs))
        seen = {(int(y), int(x)) for y, x in zip(ys, xs)}
        ends = []
        while q:
            y, x, _ = q.popleft()
            dy = ((y - cy + n // 2) % n) - n // 2
            dx = ((x - cx + n // 2) % n) - n // 2
            d = math.hypot(dy, dx)
            if d >= R - 1:
                ends.append(math.atan2(dy, dx))
                continue
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    ny, nx = (y + oy) % n, (x + ox) % n
                    if (oy or ox) and sk[ny, nx] and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        q.append((ny, nx, 0.0))
        if len(ends) < 3:
            continue
        ends.sort()
        clusters = [[ends[0]]]
        for a in ends[1:]:
            if a - clusters[-1][-1] < 0.6:
                clusters[-1].append(a)
            else:
                clusters.append([a])
        if len(clusters) > 1 and (clusters[0][0] + 2 * math.pi) - clusters[-1][-1] < 0.6:
            clusters[0] = clusters.pop() + [a + 2 * math.pi for a in clusters[0]]
        if len(clusters) != 3:
            continue
        means = sorted(float(np.mean(c)) % (2 * math.pi) for c in clusters)
        gaps = [math.degrees((means[(i + 1) % 3] - means[i]) % (2 * math.pi)) for i in range(3)]
        angles.append(min(gaps))
    return angles


# --------------------------------------------------------------------------- rendering

PURPLE_BG = np.array([228, 190, 236], dtype=np.float32)   # crystal-violet stained mucosa (RGB)
PIT_DARK = np.array([40, 10, 70], dtype=np.float32)


def render(fg: np.ndarray, phi: np.ndarray | None = None, scale: int = 1) -> np.ndarray:
    """Pit set dark on a violet mucosa; optional faint nutrient field shading."""
    h, w = fg.shape
    img = np.empty((h, w, 3), dtype=np.float32)
    shade = np.ones((h, w), dtype=np.float32)
    if phi is not None:
        shade = 0.85 + 0.15 * np.clip(phi, 0, 1)
    img[:] = PURPLE_BG * shade[..., None]
    # soften the edge with a 1-px antialias
    soft = cv2.GaussianBlur(fg.astype(np.float32), (3, 3), 0.6)
    img = img * (1 - soft[..., None]) + PIT_DARK * soft[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return img


def put_label(img: np.ndarray, lines: list[str], pad: int = 30) -> np.ndarray:
    h, w, _ = img.shape
    top = np.full((pad * len(lines) + 6, w, 3), 30, dtype=np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(top, t, (6, pad * (i + 1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([top, img])


def save_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- experiments


def run_snapshots(p: LapParams, s_list: list[float], verbose: bool = True) -> list[dict]:
    """Grow one system and record metrics/images at each s in s_list (monotone)."""
    t0 = time.time()
    sim = LaplacianPits(p)
    out = []
    for s in s_list:
        sim.run_to(s)
        fg = smooth_figure(sim.agg, p.smooth_sigma)
        m = topo_metrics(fg, roots=sim.nuclei)
        m.update(s=s, eta=round(sim.eta_now(), 3), r_w=round(sim.r_w_now(), 2), n_nuc=sim.n_nuc,
                 mass_mean=float((sim.mass[1:] - sim.mass0[1:]).mean()), t=round(time.time() - t0, 1))
        out.append(dict(metrics=m, fg=fg.copy(), phi=sim.phi.copy()))
        if verbose:
            print(json.dumps({k: v for k, v in m.items()}), flush=True)
    return out


def cmd_grid(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    etas = [float(x) for x in a.etas.split(",")]
    s_list = [float(x) for x in a.s_list.split(",")]
    rows = []
    tiles = []
    ztiles = []
    for eta in etas:
        p = LapParams(size=a.size, spacing=a.spacing, jitter=a.jitter, r0=a.r0, r_w=a.r_w, ell=a.ell,
                      eta=eta, mass_max=a.mass_max, growth=a.growth, seed=a.seed, lattice=a.lattice,
                      sor_iters=a.sor_iters, grow_frac=a.grow_frac, noise_m=a.noise_m,
                      smooth_sigma=a.smooth_sigma, beta=a.beta, beta_r=a.beta_r)
        snaps = run_snapshots(p, s_list)
        row = []
        zrow = []
        for sn in snaps:
            m = sn["metrics"]
            m["tag"] = a.tag
            rows.append(m)
            img = render(sn["fg"])
            img = put_label(img, [f"eta={eta:g} s={m['s']:.2f} cov={m['cov']:.2f}",
                                  f"Y={m['junc3']} ep={m['endpoints']} eul={m['euler']} gen={m['gen_max']}"], pad=18)
            row.append(img)
            if a.zoom:
                z = render(sn["fg"][: a.zoom, : a.zoom], scale=max(1, 288 // a.zoom))
                zrow.append(put_label(z, [f"eta={eta:g} s={m['s']:.2f} Y={m['junc3']} ep={m['endpoints']}"], pad=20))
            np.save(out / f"{a.tag}_eta{eta:g}_s{m['s']:.2f}.npy", np.packbits(sn["fg"]))
        tiles.append(np.hstack(row))
        if zrow:
            ztiles.append(np.hstack(zrow))
    grid = np.vstack(tiles)
    save_rgb(out / f"{a.tag}_grid.png", grid)
    if ztiles:
        save_rgb(out / f"{a.tag}_zoom.png", np.vstack(ztiles))
    with open(out / f"{a.tag}_metrics.json", "w") as f:
        json.dump(dict(params=asdict(p), rows=rows), f, indent=1)
    print("saved", out / f"{a.tag}_grid.png")


def cmd_path(a: argparse.Namespace) -> None:
    """One growth path s: 0 -> 1 with eta(s) and r_w(s); saves phase snapshots."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = LapParams(size=a.size, spacing=a.spacing, jitter=a.jitter, r0=a.r0, r_w=a.r_w, ell=a.ell,
                  eta=a.eta, eta_end=a.eta_end, eta_ramp=a.eta_ramp, r_w_end=a.r_w_end, mass_max=a.mass_max, growth=a.growth,
                  seed=a.seed, lattice=a.lattice, sor_iters=a.sor_iters, grow_frac=a.grow_frac, noise_m=a.noise_m,
                      smooth_sigma=a.smooth_sigma, beta=a.beta, beta_r=a.beta_r)
    s_list = [float(x) for x in a.s_list.split(",")]
    snaps = run_snapshots(p, s_list)
    rows = []
    row_imgs = []
    for sn in snaps:
        m = sn["metrics"]
        rows.append(m)
        np.save(out / f"{a.tag}_s{m['s']:.2f}.npy", np.packbits(sn["fg"]))
        img = render(sn["fg"], sn["phi"] if a.shade else None)
        img = put_label(img, [f"s={m['s']:.2f} eta={m['eta']:g} cov={m['cov']:.2f}",
                              f"Y={m['junc3']} ep={m['endpoints']} eul={m['euler']} gen={m['gen_max']}"], pad=18)
        row_imgs.append(img)
    save_rgb(out / f"{a.tag}_path.png", np.hstack(row_imgs))
    with open(out / f"{a.tag}_metrics.json", "w") as f:
        json.dump(dict(params=asdict(p), rows=rows), f, indent=1)
    print("saved", out / f"{a.tag}_path.png")


PHASE_NAMES = {
    "I": "Type I (round pits)",
    "III": "Type III (tubular pits)",
    "IVB": "Type IV branching",
    "IVV": "Type IV villous (inverted)",
}


def cmd_phases(a: argparse.Namespace) -> None:
    """Run one path and save a labelled 4-panel figure at the given phase s values."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = LapParams(size=a.size, spacing=a.spacing, jitter=a.jitter, r0=a.r0, r_w=a.r_w, ell=a.ell,
                  eta=a.eta, eta_end=a.eta_end, eta_ramp=a.eta_ramp, r_w_end=a.r_w_end, mass_max=a.mass_max,
                  growth=a.growth, seed=a.seed, lattice=a.lattice, sor_iters=a.sor_iters, grow_frac=a.grow_frac,
                  noise_m=a.noise_m, smooth_sigma=a.smooth_sigma, beta=a.beta, beta_r=a.beta_r)
    s_phase = [float(x) for x in a.s_phases.split(",")]
    keys = ["I", "III", "IVB", "IVV"]
    snaps = run_snapshots(p, s_phase)
    panels = []
    rows = []
    for key, sn in zip(keys, snaps):
        m = sn["metrics"]
        m["phase"] = key
        rows.append(m)
        img = render(sn["fg"], scale=a.scale)
        lines = [PHASE_NAMES[key],
                 f"s={m['s']:.2f} eta={m['eta']:g} cov={m['cov']:.2f}  Y={m['junc3']} X={m['junc4']} "
                 f"ep={m['endpoints']} euler={m['euler']} gen={m['gen_max']}"]
        panels.append(put_label(img, lines, pad=26))
        np.save(out / f"{a.tag}_{key}.npy", np.packbits(sn["fg"]))
        save_rgb(out / f"{a.tag}_{key}.png", render(sn["fg"], scale=a.scale))
    gap = np.full((panels[0].shape[0], 8, 3), 30, dtype=np.uint8)
    strip = panels[0]
    for pn in panels[1:]:
        strip = np.hstack([strip, gap, pn])
    save_rgb(out / f"{a.tag}_phases.png", strip)
    with open(out / f"{a.tag}_metrics.json", "w") as f:
        json.dump(dict(params=asdict(p), rows=rows), f, indent=1)
    print("saved", out / f"{a.tag}_phases.png")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grid", "path", "phases"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", default="/tmp/lap")
        sp.add_argument("--tag", default=name)
        sp.add_argument("--size", type=int, default=192)
        sp.add_argument("--spacing", type=float, default=48.0)
        sp.add_argument("--jitter", type=float, default=0.35)
        sp.add_argument("--lattice", default="hex")
        sp.add_argument("--r0", type=float, default=2.0)
        sp.add_argument("--r-w", type=float, default=1.0)
        sp.add_argument("--r-w-end", type=float, default=None)
        sp.add_argument("--ell", type=float, default=32.0)
        sp.add_argument("--mass-max", type=float, default=1200.0)
        sp.add_argument("--growth", default="per_nucleus")
        sp.add_argument("--noise-m", type=int, default=1)
        sp.add_argument("--beta", type=float, default=4.0)
        sp.add_argument("--beta-r", type=float, default=2.0)
        sp.add_argument("--smooth-sigma", type=float, default=1.0)
        sp.add_argument("--zoom", type=int, default=0, help="also save a zoomed crop (crop size px)")
        sp.add_argument("--grow-frac", type=float, default=0.25)
        sp.add_argument("--sor-iters", type=int, default=24)
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--s-list", default="0.05,0.15,0.3,0.5,0.7,1.0")
        sp.add_argument("--shade", action="store_true")
        if name == "grid":
            sp.add_argument("--etas", default="0,1,2,3")
        else:
            sp.add_argument("--eta", type=float, default=2.0)
            sp.add_argument("--eta-end", type=float, default=None)
            sp.add_argument("--eta-ramp", type=float, default=1.0)
        if name == "phases":
            sp.add_argument("--s-phases", default="0.02,0.08,0.4,0.85")
            sp.add_argument("--scale", type=int, default=2)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.cmd == "grid":
        cmd_grid(a)
    elif a.cmd == "phases":
        cmd_phases(a)
    else:
        cmd_path(a)


if __name__ == "__main__":
    main()
