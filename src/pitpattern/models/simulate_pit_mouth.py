#!/usr/bin/env python3
"""Crypt-mouth model: Round -> Elongated -> Branch -> Gyrus -> Villous from one parameter p.

Two coupled layers, every coefficient a fixed function of the progression p (0..4):

1. crypt mouths (units).  N crypts start at uniformly random positions and orientations.
   Each crypt carries one groove whose shape is set by p (dot -> tube -> Y, arm lengths from
   make_pit_grade_sweep.shape) and they relax under a soft repulsion that keeps the Type I gap D.
   For p > 2 the arm tips keep growing (self-avoiding, gap D, occasional forks) up to a length
   budget that rises with p, while the number of crypts per area falls (surface expansion).
2. mucosal field u (purple u > 0).  Gradient flow of a Swift-Hohenberg energy pinned to the
   crypt-mouth template T with strength kappa(p):

       du/dt = r u - (1 + lap / k(p)^2)^2 u - u^3 + g(p) u^2 + kappa(p) (T - u)

   kappa is large up to Branch (the field equals the units), falls to 0 through Gyrus (grooves
   join and equalise their spacing), the pitch 2 pi / k widens from Gyrus on and g < 0 breaks
   the red ridges into islands (Villous).

Rendering: the centreline of u > 0 redrawn at one fixed width (same rule for every p).

    uv run python -m pitpattern.models.simulate_pit_mouth scan --output artifacts/pit_mouth/explore --seeds 1 7 42
    uv run python -m pitpattern.models.simulate_pit_mouth sweep --output artifacts/pit_mouth/sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pitpattern.figures.make_pit_grade_sweep import PER_ARM, area_per, relax, rotate, smooth
from pitpattern.figures.make_pit_ideal_sweep import PURPLE, RED
from pitpattern.metrics.evaluate_pit_steps import prune_skeleton_spurs, zhang_suen_thin


@dataclass(frozen=True)
class Config:
    width: int = 440
    height: int = 440
    D: float = 27.5               # Type I pit spacing (px)
    dot_d: float = 14.5           # Type I pit diameter
    line_w: float = 12.5          # groove width of the unit template
    render_w: float = 13.5        # groove width of the rendered pattern (all p)
    tube_len: float = 36.0
    arm_len: float = 24.0
    relax_steps: int = 2000
    branch_pack: float = 1.35     # crypt density at Branch relative to area_per (1 -> 2 ramps in)
    # p > 2: surface expansion and tip growth
    sparse: float = 3.0           # area per crypt at p = 4 is (1 + sparse) x Branch
    spread: float = 1.0
    extra: float = 420.0          # extra groove length per crypt at full growth (px)
    fork: float = 0.12
    # field
    r: float = 0.2
    dt: float = 0.3
    steps: int = 900
    noise: float = 0.15
    kappa0: float = 1.5
    kappa_from: float = 2.2
    kappa_to: float = 2.9
    kappa_gyrus: float = 0.5      # groove-only pinning that keeps the grown grooves through Gyrus
    pin_level: float = 0.7        # value the field is held at on the grown grooves
    kappa_off: float = 3.5        # p where the pinning has vanished (Villous is free)
    lam_from: float = 2.7
    lam_gyrus: float = 1.25       # pitch at p = 3 (units of D)
    lam_villous: float = 1.8      # pitch at p = 4 (units of D)
    g_low: float = 0.6            # asymmetry up to Gyrus (> 0 keeps the red ridges connected)
    g_villous: float = -1.0
    g_from: float = 2.95
    g_gain: float = 1.2


def params(p, cfg: Config = Config()):
    """Every p-dependent coefficient (p scalar or array)."""
    p = np.asarray(p, float)
    s1 = smooth((p - cfg.kappa_from) / (cfg.kappa_to - cfg.kappa_from))
    kappa = cfg.kappa0 * (1 - s1) ** 2
    kappa_g = cfg.kappa_gyrus * s1 * (1 - smooth((p - 3.0) / (cfg.kappa_off - 3.0)))
    lam = np.interp(p, [cfg.lam_from, 3.0, 4.0], [1.0, cfg.lam_gyrus, cfg.lam_villous]) * cfg.D
    g = cfg.g_low + (cfg.g_villous - cfg.g_low) * smooth(cfg.g_gain * (p - cfg.g_from))
    expand = 1 + cfg.sparse * smooth((p - 2) / 2)
    pack = 1 + (cfg.branch_pack - 1) * smooth(p - 1)
    pe = np.clip(p, 0, 2)                                  # arm lengths follow smooth() within each stage
    pe = np.floor(pe) + smooth(pe - np.floor(pe))
    area = area_per(pe, cfg.D, cfg.tube_len, cfg.arm_len, cfg.arm_len) * expand / pack
    grow = smooth(1.4 * (p - 2))
    return dict(kappa=kappa, kappa_g=kappa_g, lam=lam, g=g, expand=expand, area=area, grow=grow)


class Hash:
    """Spatial hash of centreline points, periodic in x and y, with owner / arc / branch."""

    def __init__(self, cell, box):
        self.cell, self.box, self.d = cell, box, {}
        self.n = (int(np.ceil(box[0] / cell)), int(np.ceil(box[1] / cell)))

    def add(self, p, owner, arc, branch=-1):
        x, y = p[0] % self.box[0], p[1] % self.box[1]
        self.d.setdefault((int(x // self.cell), int(y // self.cell)), []).append((x, y, owner, arc, branch))

    def clear_of_foreign(self, p, owner, arc, skip, radius, core=0.0, anc=None):
        """True if no foreign groove point lies within `radius` of p."""
        x, y = p[0] % self.box[0], p[1] % self.box[1]
        cx, cy = int(x // self.cell), int(y // self.cell)
        k = int(np.ceil(radius / self.cell))
        for gx in range(cx - k, cx + k + 1):
            for gy in range(cy - k, cy + k + 1):
                for qx, qy, o, a, b in self.d.get((gx % self.n[0], gy % self.n[1]), ()):
                    if o == owner:
                        if a < core:
                            continue
                        if anc is not None and b in anc and a <= anc[b] and abs(a - arc) < skip:
                            continue
                    dx = qx - x
                    dx -= self.box[0] * round(dx / self.box[0])
                    dy = qy - y
                    dy -= self.box[1] * round(dy / self.box[1])
                    if dx * dx + dy * dy < radius * radius:
                        return False
        return True


def place_units(px, rng, cfg: Config):
    """Crypt units for the progression profile px (one value per column): centres, segments."""
    W, H = cfg.width, cfg.height
    box = np.array([W, H], float)
    D, L, l = cfg.D, cfg.tube_len, cfg.arm_len
    xs = np.arange(W) + 0.5
    dens = 1 / params(px, cfg)["area"]
    n = max(1, int(round(dens.sum() * H)))
    cdf = np.cumsum(dens) / dens.sum()
    c = np.stack([np.interp(rng.random(n), cdf, xs), rng.uniform(0, H, n)], 1)
    q = px[np.clip(c[:, 0].astype(int), 0, W - 1)]
    dscale = 1 + cfg.spread * (np.sqrt(params(q, cfg)["expand"]) - 1)
    c, th, local = relax(c, np.minimum(q, 2), D, L, l, l, rng, box, steps=cfg.relax_steps, dscale=dscale)
    pts = rotate(local, th) + c[:, None]

    hsh = Hash(1.3 * D, box)
    segs, tips = [], []
    for i in range(n):
        hsh.add(c[i], i, 0.0)
        for arm in range(3):
            tip = pts[i, (arm + 1) * PER_ARM]
            v = tip - c[i]
            ln = np.linalg.norm(v)
            if ln < 0.5:
                continue
            segs.append((c[i].copy(), tip.copy()))
            bid = 3 * i + arm
            for f in np.linspace(0.25, 1, 4):
                hsh.add(c[i] + f * v, i, f * ln, bid)
            g = float(params(q[i], cfg)["grow"])
            if g > 0.02:
                sib = {3 * i + k: l + D / np.sqrt(3) for k in range(3)}
                sib[bid] = np.inf
                tips.append([i, tip.copy(), np.arctan2(v[1], v[0]), ln, g * cfg.extra / 3, bid, sib])

    # tip growth: self-avoiding, keeps the centreline distance D from every other groove
    step, next_bid = 2.0, 3 * n
    gap = params(q, cfg)["lam"]                      # centreline distance kept by crypt i's tips
    active = tips
    while active:
        nxt = []
        for t in rng.permutation(len(active)):
            o, p, h, arc, bud, bid, anc = active[t]
            if bud <= 0:
                continue
            g = float(smooth(q[o] - 2))
            for dh in sorted([0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.25, -1.25], key=lambda v: abs(v) + 0.2 * rng.random()):
                hh = h + dh + rng.normal(0, 0.08)
                np_ = p + step * np.array([np.cos(hh), np.sin(hh)])
                core = l + 1 if arc < l + gap[o] else 0.0
                if hsh.clear_of_foreign(np_, o, arc + step, skip=1.05 * gap[o], radius=gap[o], core=core, anc=anc):
                    segs.append((p.copy(), np_.copy()))
                    hsh.add(np_, o, arc + step, bid)
                    nxt.append([o, np_, hh, arc + step, bud - step, bid, anc])
                    if rng.random() < (cfg.fork + 0.10 * (abs(dh) > 0.8)) * g:
                        side = hh + rng.choice([-1, 1]) * np.pi / 3
                        near = arc + step + 1.3 * gap[o]
                        nanc = dict(anc)
                        nanc[bid] = near
                        nanc[next_bid] = np.inf
                        anc[next_bid] = near
                        nxt.append([o, np_.copy(), side, arc + step, (bud - step) * 0.6, next_bid, nanc])
                        next_bid += 1
                    break
        active = nxt
    return c, segs


def draw_template(c, segs, cfg: Config, ss: int = 3):
    """Units drawn as dots and constant-width lines on the periodic box -> coverage in [0, 1]."""
    W, H = cfg.width, cfg.height
    m = np.zeros((H * ss, W * ss), np.uint8)
    P = lambda v: (int(round(v[0] * ss)), int(round(v[1] * ss)))
    for ox in (-W, 0, W):
        for oy in (-H, 0, H):
            off = np.array([ox, oy], float)
            for ci in c:
                v = ci + off
                if -40 < v[0] < W + 40 and -40 < v[1] < H + 40:
                    cv2.circle(m, P(v), int(cfg.dot_d / 2 * ss), 1, -1, cv2.LINE_AA)
            for s0, s1 in segs:
                a, b = s0 + off, s1 + off
                if (-40 < a[0] < W + 40 and -40 < a[1] < H + 40) or (-40 < b[0] < W + 40 and -40 < b[1] < H + 40):
                    cv2.line(m, P(a), P(b), 1, int(cfg.line_w * ss), cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)


def blur_wrap(f, sigma):
    pad = int(4 * sigma) + 1
    return cv2.GaussianBlur(np.pad(f, pad, mode="wrap"), (0, 0), sigma)[pad:-pad, pad:-pad]


class Field:
    """Pinned Swift-Hohenberg gradient flow on the periodic box (semi-implicit spectral steps)."""

    def __init__(self, T, px, rng, cfg: Config):
        H, W = T.shape
        self.cfg, self.T = cfg, T
        pr = params(px, cfg)
        self.kappa = pr["kappa"][None, :]
        self.kappa_g = pr["kappa_g"][None, :] * np.clip((T + 1) / 2, 0, 1)     # acts on the grooves only
        self.g = pr["g"][None, :]
        K2 = (2 * np.pi * np.fft.fftfreq(H))[:, None] ** 2 + (2 * np.pi * np.fft.fftfreq(W))[None, :] ** 2
        den = lambda lam: 1 - cfg.dt * (cfg.r - (1 - K2 / (2 * np.pi / lam) ** 2) ** 2 - 1)
        lam = pr["lam"]
        if np.ptp(lam) < 1e-9:                       # uniform p: exact pitch
            self.dens, self.wts = [den(float(lam[0]))], [1.0]
        else:                                        # gradient: blend of constant-pitch steps
            levels = np.array([1.0, cfg.lam_gyrus, 0.5 * (cfg.lam_gyrus + cfg.lam_villous), cfg.lam_villous]) * cfg.D
            self.dens = [den(L_) for L_ in levels]
            self.wts = [np.interp(lam, levels, np.eye(len(levels))[i])[None, :] for i in range(len(levels))]
        self.u = cfg.noise * rng.normal(size=T.shape)      # the field itself starts from noise

    def step(self, n=1):
        cfg, u = self.cfg, self.u
        for _ in range(n):
            F = np.fft.fft2(u) + cfg.dt * np.fft.fft2(u - u ** 3 + self.g * u * u + self.kappa * (self.T - u)
                                                   + self.kappa_g * (cfg.pin_level - u))
            u = sum(w * np.fft.ifft2(F / d).real for w, d in zip(self.wts, self.dens))
        self.u = u
        return u


def centreline(binary, min_area=0.0, periodic=True):
    """Skeleton of a (periodic) binary image; a blob too round to keep a skeleton gives one pixel."""
    pad = 24
    b = np.pad(binary, pad, mode="wrap" if periodic else "constant")
    sk = prune_skeleton_spurs(zhang_suen_thin(b), min_length=4).astype(bool)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    has = np.zeros(n, bool)
    has[np.unique(lab[sk])] = True
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            sk[lab == i] = False
        elif not has[i]:
            sk[int(round(cen[i, 1])), int(round(cen[i, 0]))] = True
    return sk[pad:-pad, pad:-pad]


def render(u, cfg: Config, ss: int = 3):
    """Centreline of u > 0 redrawn at the fixed width -> (rgb, binary)."""
    H, W = u.shape
    sk = centreline(u > 0, min_area=0.3 * cfg.line_w ** 2)
    m = np.zeros((H * ss, W * ss), np.uint8)
    r = int(round(cfg.render_w / 2 * ss))
    for y, x in np.argwhere(sk):
        for ox in ((-W, 0, W) if x < 12 or x > W - 12 else (0,)):
            for oy in ((-H, 0, H) if y < 12 or y > H - 12 else (0,)):
                cv2.circle(m, (int((x + 0.5 + ox) * ss), int((y + 0.5 + oy) * ss)), r, 1, -1)
    cover = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    binary = cover > 0.5
    return colour(cover, cfg), binary


def colour(cover, cfg: Config):
    """Soft slide-6 look: blur and map to red -> purple (same rule as the answer image)."""
    phi = blur_wrap(cover, 4.0)
    ref = np.zeros((61, 61), np.float32)
    cv2.circle(ref, (30, 30), int(round(cfg.dot_d / 2)), 1.0, -1, cv2.LINE_AA)
    peak = cv2.GaussianBlur(ref, (0, 0), 4.0).max()
    x = np.clip((phi / peak - 0.02) / 0.98, 0, 1)
    t = (x * x * (3 - 2 * x))[..., None]
    return np.clip(RED * (1 - t) + PURPLE * t, 0, 255).astype(np.uint8)


def run(px, seed: int, cfg: Config = Config(), holds=(1,)):
    """Simulate the profile px; returns a list of (multiple of cfg.steps, u) and the unit template."""
    rng = np.random.default_rng(seed)
    px = np.broadcast_to(np.asarray(px, float), (cfg.width,)).copy()
    c, segs = place_units(px, rng, cfg)
    T = 2 * blur_wrap(draw_template(c, segs, cfg), 1.5) - 1
    fld = Field(T, px, np.random.default_rng(seed + 10_000), cfg)
    out, done = [], 0
    for h in sorted(holds):
        fld.step(int(h * cfg.steps) - done)
        done = int(h * cfg.steps)
        out.append((h, fld.u.copy()))
    return out, dict(n_crypts=len(c), n_segments=len(segs), T=T)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="independent uniform-p runs")
    sc.add_argument("--output", type=Path, required=True)
    sc.add_argument("--seeds", type=int, nargs="+", default=[1])
    sc.add_argument("--ps", type=float, nargs="+", required=True)
    sc.add_argument("--holds", type=float, nargs="+", default=[1])
    sc.add_argument("--set", nargs="*", default=[], help="Config overrides key=value")
    sw = sub.add_parser("sweep", help="one wide field, p rising left to right")
    sw.add_argument("--output", type=Path, required=True)
    sw.add_argument("--seed", type=int, default=1)
    sw.add_argument("--width", type=int, default=1002)
    sw.add_argument("--height", type=int, default=497)
    sw.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = Config()
    over = {k: type(getattr(cfg, k))(v) for k, v in (s.split("=") for s in a.set)}
    cfg = replace(cfg, **over)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    if a.cmd == "scan":
        from pitpattern.evaluation.evaluate_pit_mouth import label, measure
        rows = []
        for seed in a.seeds:
            for p in a.ps:
                states, info = run(p, seed, cfg, holds=a.holds)
                first = None
                for h, u in states:
                    rgb, binary = render(u, cfg)
                    tag = f"p{p:.2f}_seed{seed}" + ("" if h == min(a.holds) else f"_x{h:g}")
                    Image.fromarray(rgb).save(a.output / f"{tag}.png")
                    np.save(a.output / f"{tag}.npy", binary)
                    row = dict(p=p, seed=seed, hold=h, n_crypts=info["n_crypts"], **measure(binary, cfg))
                    row["label"] = label(row)
                    row["raw_width_median"] = float(np.median(cv2.distanceTransform((u > 0).astype(np.uint8), cv2.DIST_L2, 5)[centreline(u > 0)]) * 2) if (u > 0).any() else 0.0
                    if first is None:
                        first = binary
                    row["changed_fraction"] = float((binary != first).mean())
                    rows.append(row)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        with open(a.output / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    else:
        pad = 60
        cfg = replace(cfg, width=a.width + 2 * pad, height=a.height)
        uu = (np.arange(cfg.width) - pad) / a.width
        px = np.interp(uu, [0.0, 0.03, 0.20, 0.23, 0.31, 0.43, 0.57, 0.73, 0.92, 1.0], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        states, info = run(px, a.seed, cfg)
        rgb, binary = render(states[-1][1], cfg)
        Image.fromarray(rgb[:, pad:-pad]).save(a.output / "sweep.png")
        np.save(a.output / "sweep_binary.npy", binary[:, pad:-pad])
        print("saved", a.output / "sweep.png", "crypts", info["n_crypts"])


if __name__ == "__main__":
    main()
