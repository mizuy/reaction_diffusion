#!/usr/bin/env python3
"""Crypt lobulation model: pits join into bifid tubes, trifid Ys and finally a network.

Biological reading
------------------
Each pit seen from the surface is a lobe of a crypt opening.  Lobes sit on the
near-hexagonal lattice of crypt niches.  With progression p a crypt gathers
more lobes: two lobes joined through the crypt lumen give a straight tube
(IIIL), three lobes joined at a common lumen give a Y (IV_B), and when crypt
lumens fuse with all their neighbours the grooves form a network that leaves
the mucosa as round islands (IV_V, the light/dark inverse of Type I).

Model
-----
1. Lobe positions: N particles start at independent random positions in a
   periodic box and relax under soft repulsion into a near-hexagonal lattice
   (Type I).  Nothing about the lattice is imposed beyond the mean spacing.
2. Candidate units on that lattice: every nearest-neighbour edge (a straight
   two-lobe tube) and every triangle of mutual neighbours (a three-lobe Y whose
   arms meet at the triangle centre).
3. Energy, with all coefficients fixed piecewise-linear functions of p only:
       E = - sum_tubes mu_e(p) - sum_Ys mu_t(p)
           + sum_lobes [ c0(p) if unjoined ;  K(p) (v - 1) if in v >= 1 units ]
           + J * (number of active units that would overwrite the mucosa between
                  them: two Ys sharing an edge, a tube and a Y on the same edge)
   mu_e: gain of a bifid crypt (rises, then is replaced by trifid crypts),
   mu_t: gain of a trifid crypt, K: barrier against a lobe belonging to more
   than one crypt (crypt fusion), c0: cost of an unjoined lobe.
4. Dynamics: Metropolis updates (single flips, replacement of the units using
   the same lobes, and swaps with conflicting units) from an independent random
   initial state, with simulated annealing.  Which lobes pair, the orientation
   of each tube and Y, and the IV_V domain orientation are decided locally.

Valence (how many crypts a lobe belongs to) is the assumed ingredient: the
model does not explain why it changes with p, only what follows from it.

    uv run python -m pitpattern.models.simulate_pit_crypt scan --output artifacts/pit_crypt/try --seeds 1 2 3
    uv run python -m pitpattern.models.simulate_pit_crypt sweep --output artifacts/pit_crypt/sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PURPLE = np.array([81, 10, 109], np.float32)  # colours of slide 6 (goal_rd_sweep_i_to_iv.png)
RED = np.array([216, 10, 3], np.float32)


@dataclass(frozen=True)
class Config:
    spacing: float = 27.5     # lobe spacing in px (slide 6: nearest-neighbour 27.4 px)
    cols: int = 9             # box = cols x rows lattice cells (periodic)
    rows: int = 10
    relax_steps: int = 600
    sweeps: int = 500
    T1: float = 0.02
    J: float = 50.0
    init_frac: float = 0.15   # fraction of units switched on in the random initial state
    dot_d: float = 14.5       # drawn lobe diameter (px)
    line_w: float = 10.5      # drawn tube / arm width (px): blurred width matches a blurred lobe
    blur: float = 4.0

    @property
    def box(self) -> np.ndarray:
        return np.array([self.cols * self.spacing, self.rows * self.spacing * np.sqrt(3) / 2])

    @property
    def n_lobes(self) -> int:
        return self.cols * self.rows


def params(p: float) -> tuple[float, float, float, float]:
    """(mu_e, mu_t, K, c0): fixed piecewise-linear functions of the single control p."""
    mu_e = 2.0 * (p - 0.5) - 4.0 * max(0.0, p - 1.6)
    mu_t = 3.4 * (p - 1.05)
    K = max(0.0, 6.0 * (2.9 - p))
    c0 = max(0.0, 1.5 * (p - 0.5))
    return mu_e, mu_t, K, c0


# ---------------------------------------------------------------- lattice
def wrap(d: np.ndarray, L: np.ndarray) -> np.ndarray:
    return d - L * np.round(d / L)


def relax(n: int, L: np.ndarray, a: float, rng, steps: int = 600, noise: float = 1.5) -> np.ndarray:
    """Soft-repulsion relaxation from uniformly random positions (periodic box)."""
    x = rng.uniform(0, 1, (n, 2)) * L
    for it in range(steps):
        d = wrap(x[:, None, :] - x[None, :, :], L)
        r = np.sqrt((d ** 2).sum(-1)) + np.eye(n) * 1e9
        f = np.clip(1.15 * a - r, 0, None) / r
        x += 0.25 * (d * f[..., None]).sum(1)
        x += rng.normal(0, noise * (1 - it / steps), x.shape)
        x %= L
    return x


class Lattice:
    def __init__(self, x: np.ndarray, L: np.ndarray, a: float):
        self.x, self.L, self.a = x, L, a
        n = len(x)
        r = np.sqrt((wrap(x[:, None, :] - x[None, :, :], L) ** 2).sum(-1))
        nb = (r < 1.3 * a) & (r > 0)
        self.nbrs = [set(np.flatnonzero(nb[i]).tolist()) for i in range(n)]
        self.edges = sorted({(i, j) for i in range(n) for j in self.nbrs[i] if i < j})
        self.edge_id = {e: k for k, e in enumerate(self.edges)}
        tris = set()
        for i, j in self.edges:
            for k in self.nbrs[i] & self.nbrs[j]:
                tris.add(tuple(sorted((i, j, k))))
        self.tris = sorted(tris)
        self.tri_c = np.array([x[t[0]] + np.mean([np.zeros(2)] + [wrap(x[v] - x[t[0]], L) for v in t[1:]], 0)
                               for t in self.tris]).reshape(-1, 2) % L
        self.edge_tris: dict[tuple[int, int], list[int]] = {e: [] for e in self.edges}
        for ti, t in enumerate(self.tris):
            for e in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2])):
                self.edge_tris[e].append(ti)
        self.tri_adj = [set() for _ in self.tris]
        for ts in self.edge_tris.values():
            for a_ in ts:
                self.tri_adj[a_].update(b for b in ts if b != a_)


class Units:
    """Tubes (edges) and Ys (triangles) in one index space, with their conflicts."""

    def __init__(self, lat: Lattice):
        ne = len(lat.edges)
        self.ne = ne
        self.verts = [list(e) for e in lat.edges] + [list(t) for t in lat.tris]
        self.is_tri = np.array([False] * ne + [True] * len(lat.tris))
        mids = [lat.x[i] + 0.5 * wrap(lat.x[j] - lat.x[i], lat.L) for i, j in lat.edges]
        self.pos = np.array(mids + list(lat.tri_c)) % lat.L
        conf = [set() for _ in self.verts]
        for ti, t in enumerate(lat.tris):
            u = ne + ti
            conf[u].update(ne + v for v in lat.tri_adj[ti])
            for e in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2])):
                k = lat.edge_id[e]
                conf[u].add(k)
                conf[k].add(u)
        self.conf = [sorted(c) for c in conf]
        self.pit_units = [[] for _ in lat.x]
        for u, vs in enumerate(self.verts):
            for v in vs:
                self.pit_units[v].append(u)


# ---------------------------------------------------------------- dynamics
class CryptModel:
    def __init__(self, lat: Lattice, p_field, cfg: Config):
        """p_field(position) -> p; a constant function gives a uniform run."""
        self.lat, self.cfg = lat, cfg
        self.U = U = Units(lat)
        pu = np.array([params(p_field(c)) for c in U.pos])
        self.gain = np.where(U.is_tri, pu[:, 1], pu[:, 0])
        self.pp = np.array([params(p_field(c)) for c in lat.x])
        self.on = np.zeros(len(U.verts), bool)
        self.val = np.zeros(len(lat.x), int)

    def randomise(self, rng) -> None:
        self.on = rng.random(len(self.U.verts)) < self.cfg.init_frac
        self.val[:] = 0
        for u in np.flatnonzero(self.on):
            self.val[self.U.verts[u]] += 1

    def _pit_cost(self, i: int, v: int) -> float:
        K, c0 = self.pp[i, 2], self.pp[i, 3]
        return c0 if v == 0 else K * (v - 1)

    def delta(self, toggle: list[int]) -> float:
        on, U, J = self.on, self.U, self.cfg.J
        dE, dv, tset = 0.0, {}, set(toggle)
        for u in toggle:
            s = -1 if on[u] else 1
            dE -= s * self.gain[u]
            for v in U.verts[u]:
                dv[v] = dv.get(v, 0) + s
        for v, d in dv.items():
            dE += self._pit_cost(v, self.val[v] + d) - self._pit_cost(v, self.val[v])
        for u in toggle:
            new_u = not on[u]
            for w in U.conf[u]:
                if w in tset and w < u:
                    continue
                new_w = (not on[w]) if w in tset else on[w]
                dE += J * (int(new_u and new_w) - int(on[u] and on[w]))
        return dE

    def apply(self, toggle: list[int]) -> None:
        for u in toggle:
            s = -1 if self.on[u] else 1
            self.on[u] = not self.on[u]
            for v in self.U.verts[u]:
                self.val[v] += s

    def sweep(self, rng, T: float) -> int:
        U, on = self.U, self.on
        nu, changed = len(U.verts), 0
        for _ in range(nu):
            u = int(rng.integers(nu))
            r = rng.random()
            if on[u] or r < 0.4:
                toggle = [u]
            elif r < 0.7:   # open u, close the units that use one of its lobes
                toggle = [u] + sorted({w for v in U.verts[u] for w in U.pit_units[v] if on[w] and w != u})
            else:           # open u, close the units it conflicts with
                toggle = [u] + [w for w in U.conf[u] if on[w]]
            dE = self.delta(toggle)
            if dE < 0 or rng.random() < np.exp(-dE / T):
                self.apply(toggle)
                changed += 1
        return changed

    def anneal(self, rng, sweeps: int | None = None) -> None:
        sweeps = self.cfg.sweeps if sweeps is None else sweeps
        T0 = max(2.0, float(np.abs(self.gain).max()))  # hot relative to the largest binding energy
        for s in range(sweeps):
            self.sweep(rng, T0 * (self.cfg.T1 / T0) ** (s / max(sweeps - 1, 1)))

    def hold(self, rng, sweeps: int) -> None:
        for _ in range(sweeps):
            self.sweep(rng, self.cfg.T1)

    @property
    def tubes(self) -> np.ndarray:
        return self.on[:self.U.ne]

    @property
    def ys(self) -> np.ndarray:
        return self.on[self.U.ne:]


# ---------------------------------------------------------------- rendering and measurement
def render(lat: Lattice, tubes, ys, cfg: Config, size: tuple[int, int] | None = None, ss: int = 3):
    """Slide-6 look: blur the drawn lobes/tubes/arms and map intensity to red -> purple."""
    L = lat.L
    W, H = size if size else (int(round(L[0])), int(round(L[1])))
    m = np.zeros((H * ss, W * ss), np.uint8)

    def P(pt):
        return int(round(pt[0] * ss)), int(round(pt[1] * ss))

    rd, rw = int(cfg.dot_d / 2 * ss), int(cfg.line_w * ss)
    for sx in (-L[0], 0, L[0]):
        for sy in (-L[1], 0, L[1]):
            sh = np.array([sx, sy])
            for v in lat.x:
                cv2.circle(m, P(v + sh), rd, 1, -1, cv2.LINE_AA)
            for k in np.flatnonzero(tubes):
                i, j = lat.edges[k]
                cv2.line(m, P(lat.x[i] + sh), P(lat.x[i] + wrap(lat.x[j] - lat.x[i], L) + sh), 1, rw, cv2.LINE_AA)
            for t in np.flatnonzero(ys):
                c = lat.tri_c[t]
                for v in lat.tris[t]:
                    cv2.line(m, P(c + sh), P(c + wrap(lat.x[v] - c, L) + sh), 1, rw, cv2.LINE_AA)
                cv2.circle(m, P(c + sh), rw // 2, 1, -1, cv2.LINE_AA)
    small = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    phi = cv2.GaussianBlur(small, (0, 0), cfg.blur)
    ref = np.zeros((61, 61), np.float32)
    cv2.circle(ref, (30, 30), int(round(cfg.dot_d / 2)), 1.0, -1, cv2.LINE_AA)
    peak = cv2.GaussianBlur(ref, (0, 0), cfg.blur).max()
    xq = np.clip((phi / peak - 0.02) / 0.98, 0, 1)
    tq = xq * xq * (3 - 2 * xq)
    rgb = RED * (1 - tq[..., None]) + PURPLE * tq[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8), phi > 0.5 * peak


def composition(model: CryptModel) -> dict[str, float]:
    n = len(model.lat.x)
    h = np.bincount(model.val, minlength=4)
    n_t, n_y = int(model.tubes.sum()), int(model.ys.sum())
    in_tube = np.zeros(n, bool)
    in_y = np.zeros(n, bool)
    for k in np.flatnonzero(model.tubes):
        in_tube[list(model.lat.edges[k])] = True
    for t in np.flatnonzero(model.ys):
        in_y[list(model.lat.tris[t])] = True
    return dict(n_lobes=n, tubes=n_t, ys=n_y, v0=int(h[0]), v1=int(h[1]), v2=int(h[2]), v3=int(h[3:].sum()),
                frac_single=float(h[0] / n), frac_tube_only=float((in_tube & ~in_y & (model.val == 1)).mean()),
                frac_y_only=float((in_y & ~in_tube & (model.val == 1)).mean()), frac_v3=float(h[3:].sum() / n))


def label(c: dict[str, float]) -> str:
    """Band label from lobe composition (thresholds fixed on exploration seeds 1-3)."""
    if c["frac_single"] >= 0.9:
        return "I"
    if c["frac_tube_only"] >= 0.9:
        return "IIIL"
    if c["frac_y_only"] + c["frac_tube_only"] >= 0.9 and c["frac_y_only"] >= 0.75:
        return "IV_B"
    if c["frac_v3"] >= 0.85:
        return "IV_V"
    return "transition"


def image_metrics(binary: np.ndarray, spacing: float) -> dict[str, float]:
    from pitpattern.metrics.pit_fork_metrics import fork_metrics
    from pitpattern.metrics.pit_goal_metrics import measure_goal_pattern

    m = measure_goal_pattern(binary.astype(np.float32))
    keep = ("binary_coverage", "feature_width_median", "feature_width_p90", "longest_component_fraction",
            "background_island_count", "component_count")
    out = {k: float(m[k]) for k in keep}
    out.update({k: float(v) for k, v in fork_metrics(binary, arm_min=spacing / 4).items()
                if k in ("fork_count", "fork_components", "unit_count")})
    # Opening diameter of round pits: the skeleton width estimate collapses for disks
    # (their skeleton is pruned away), so Type I is better described by the area-equivalent diameter.
    n, _, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    out["equiv_diameter_median"] = float(np.median(2 * np.sqrt(areas / np.pi))) if len(areas) else 0.0
    return out


# ---------------------------------------------------------------- CLI
def run_one(cfg: Config, p: float, seed: int, hold: tuple[int, ...] = ()):
    rng = np.random.default_rng(seed)
    lat = Lattice(relax(cfg.n_lobes, cfg.box, cfg.spacing, rng, cfg.relax_steps), cfg.box, cfg.spacing)
    model = CryptModel(lat, lambda pos: p, cfg)
    model.randomise(rng)
    model.anneal(rng)
    snaps = {0: model.on.copy()}
    done = 0
    for h in sorted(hold):
        model.hold(rng, h - done)
        done = h
        snaps[h] = model.on.copy()
    return lat, model, snaps


def scan(cfg: Config, ps, seeds, out: Path, hold=()) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows, grid = [], []
    for seed in seeds:
        imgs = []
        for p in ps:
            lat, model, snaps = run_one(cfg, p, seed, hold)
            rgb, binary = render(lat, model.tubes, model.ys, cfg)
            rec = dict(p=float(p), seed=int(seed), **composition(model), **image_metrics(binary, cfg.spacing))
            rec["label"] = label(rec)
            for h, st in snaps.items():
                if h:
                    rec[f"units_changed_after_{h}"] = int((st != snaps[0]).sum())
            rows.append(rec)
            imgs.append(rgb)
            Image.fromarray(rgb).save(out / f"seed{seed}_p{p:.2f}.png")
            print(json.dumps({k: rec[k] for k in ("p", "seed", "label", "tubes", "ys", "v0", "v1", "v2", "v3")}), flush=True)
        grid.append(imgs)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "config.json").write_text(json.dumps(dict(config=asdict(cfg), ps=list(ps), seeds=list(seeds),
                                                     hold=list(hold)), indent=2))
    sheet = Image.new("RGB", (len(ps) * (grid[0][0].shape[1] + 6), len(seeds) * (grid[0][0].shape[0] + 22)), "white")
    d = ImageDraw.Draw(sheet)
    for i, imgs in enumerate(grid):
        for j, im in enumerate(imgs):
            x, y = j * (im.shape[1] + 6), i * (im.shape[0] + 22)
            sheet.paste(Image.fromarray(im), (x, y))
            d.text((x + 2, y + im.shape[0] + 4), f"seed {seeds[i]}  p={ps[j]:.2f}", fill="black")
    sheet.save(out / "sheet.png")
    return rows


def spatial_sweep(cfg: Config, out: Path, seed: int, p_left: float, p_right: float, cols: int, rows: int) -> None:
    """One field with p varying linearly from left to right (for a slide-6 style picture)."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, cols=cols, rows=rows)
    rng = np.random.default_rng(seed)
    L = cfg.box
    lat = Lattice(relax(cfg.n_lobes, L, cfg.spacing, rng, cfg.relax_steps), L, cfg.spacing)
    model = CryptModel(lat, lambda pos: p_left + (p_right - p_left) * pos[0] / L[0], cfg)
    model.randomise(rng)
    model.anneal(rng)
    rgb, _ = render(lat, model.tubes, model.ys, cfg)
    Image.fromarray(rgb).save(out / f"sweep_seed{seed}.png")
    (out / f"sweep_seed{seed}.json").write_text(json.dumps(dict(config=asdict(cfg), seed=seed, p_left=p_left,
                                                                p_right=p_right, **composition(model)), indent=2))
    print("saved", out / f"sweep_seed{seed}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="independent runs at uniform p")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--ps", type=float, nargs="+", default=[0.3, 0.9, 2.3, 3.3])
    s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument("--hold", type=int, nargs="*", default=[], help="extra low-temperature sweeps to record")
    s.add_argument("--sweeps", type=int, default=Config.sweeps)
    s.add_argument("--cols", type=int, default=Config.cols)
    s.add_argument("--rows", type=int, default=Config.rows)
    w = sub.add_parser("sweep", help="one field with p increasing left to right")
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--seed", type=int, default=1)
    w.add_argument("--p-left", type=float, default=0.0)
    w.add_argument("--p-right", type=float, default=3.6)
    w.add_argument("--cols", type=int, default=36)
    w.add_argument("--rows", type=int, default=21)
    w.add_argument("--sweeps", type=int, default=Config.sweeps)
    a = ap.parse_args()
    if a.cmd == "scan":
        cfg = replace(Config(), sweeps=a.sweeps, cols=a.cols, rows=a.rows)
        scan(cfg, a.ps, a.seeds, a.output, tuple(a.hold))
    else:
        spatial_sweep(replace(Config(), sweeps=a.sweeps), a.output, a.seed, a.p_left, a.p_right, a.cols, a.rows)


if __name__ == "__main__":
    main()
