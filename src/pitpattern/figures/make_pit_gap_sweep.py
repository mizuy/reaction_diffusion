#!/usr/bin/env python3
"""Answer-image candidate with a constant gap between pits (drawn target, not a simulation).

Rule: the red mucosa between any two pit structures keeps the same width as the gap
between Type I pits.  Each crypt carries one pit shape -- a dot (I), a straight tube
(IIIL) or a three-armed Y (IV_B) -- and crypts are packed by relaxing position AND
orientation under a soft repulsion between their skeletons (target skeleton distance D
= Type I spacing), starting from random positions and orientations.  Crypt spacing
therefore grows by itself where the shapes are larger.  IV_V is the light/dark inverse
of Type I: island particles on the same spacing, purple everywhere else.

    uv run python -m pitpattern.figures.make_pit_gap_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pitpattern.figures.make_pit_ideal_sweep import PURPLE, RED, font

ROOT = Path(__file__).resolve().parents[3]
NPTS = 13


def skeleton(kind: int, L_tube: float, l_arm: float) -> np.ndarray:
    """NPTS local skeleton points (padded by repeating the centre) for kind 0/1/2/3."""
    if kind in (0, 3):
        pts = np.zeros((1, 2))
    elif kind == 1:
        t = np.linspace(-L_tube / 2, L_tube / 2, 7)
        pts = np.stack([t, np.zeros_like(t)], 1)
    else:
        pts = [np.zeros(2)]
        for k in range(3):
            a = 2 * np.pi * k / 3
            pts += [s * np.array([np.cos(a), np.sin(a)]) for s in np.linspace(l_arm / 4, l_arm, 4)]
        pts = np.array(pts)
    out = np.zeros((NPTS, 2))
    out[:len(pts)] = pts
    out[len(pts):] = pts[0]
    return out


def rotate(local: np.ndarray, th: np.ndarray) -> np.ndarray:
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    return np.stack([local[..., 0] * c - local[..., 1] * s, local[..., 0] * s + local[..., 1] * c], -1)


def relax(c, kinds, D, L_tube, l_arm, rng, box, steps=1600, k=0.25, noise=1.0, max_step=1.0):
    n = len(c)
    th = rng.uniform(0, 2 * np.pi, n)
    local = np.stack([skeleton(int(kd), L_tube, l_arm) for kd in kinds])
    reach = np.abs(local).max(axis=(1, 2))
    npts = np.array([1 if kd in (0, 3) else (7 if kd == 1 else 13) for kd in kinds], float)
    inert = np.maximum((local ** 2).sum(axis=(1, 2)) / (NPTS / npts), 1.0)
    I = J = None
    for it in range(steps):
        if it % 25 == 0:
            d = c[:, None] - c[None]
            d -= box * np.round(d / box)
            r = np.sqrt((d ** 2).sum(-1))
            I, J = np.nonzero(np.triu(r < D + reach[:, None] + reach[None] + 6, 1))
        W = rotate(local, th)
        sep = c[I] - c[J]
        sep -= box * np.round(sep / box)
        Pi = W[I] + sep[:, None]
        dv = Pi[:, :, None] - W[J][:, None]
        dist = np.sqrt((dv ** 2).sum(-1)) + 1e-9
        f = (k * np.clip(D - dist, 0, None) / dist)[..., None] * dv
        # padded duplicates of the centre would multiply the centre force: weight them down
        wi = 1.0 / np.array([NPTS - npts[x] + 1 for x in range(n)])
        fi = f.sum(2)
        fj = -f.sum(1)
        F = np.zeros((n, 2))
        T = np.zeros(n)
        np.add.at(F, I, fi.sum(1))
        np.add.at(F, J, fj.sum(1))
        np.add.at(T, I, (W[I][..., 0] * fi[..., 1] - W[I][..., 1] * fi[..., 0]).sum(1))
        np.add.at(T, J, (W[J][..., 0] * fj[..., 1] - W[J][..., 1] * fj[..., 0]).sum(1))
        amp = noise * max(0.0, 1 - it / (0.7 * steps))
        step = F / np.sqrt(NPTS)
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, max_step / np.maximum(norm, 1e-9))
        c = (c + step + rng.normal(0, amp, c.shape)) % box
        th += np.clip(2.0 * T / inert, -0.05, 0.05) + rng.normal(0, 0.05 * amp, n) * (kinds == 1) + \
            rng.normal(0, 0.05 * amp, n) * (kinds == 2)
    return c, th, local


def area_per(kind, D, L_tube, l_arm):
    return {0: D * D * 0.866, 1: (D + L_tube) * D * 0.866, 2: (D + 1.25 * l_arm) ** 2 * 0.866, 3: D * D * 0.866}[kind]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/images/pit_gap_sweep.png")
    ap.add_argument("--compare", type=Path, default=ROOT / "docs/images/pit_gap_sweep_compare.png")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1600)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Wv, Hv, pad = 1002, 497, 60
    box = np.array([Wv + 2 * pad, Hv], float)       # periodic in y; x padded and cropped
    D, dot_d, line_w = 27.5, 14.5, 12.5
    L_tube, l_arm = 20.0, 13.0
    edges = np.array([0.25, 0.50, 0.75]) * Wv + pad  # band borders (x in box coords)
    fill = {0: 0.97, 1: 0.80, 2: 0.76, 3: 0.97}
    c, kinds = [], []
    bounds = [0, *edges, box[0]]
    for kd in range(4):
        x0, x1 = bounds[kd], bounds[kd + 1]
        n = int((x1 - x0) * Hv / area_per(kd, D, L_tube, l_arm) * fill[kd])
        c.append(np.stack([rng.uniform(x0, x1, n), rng.uniform(0, Hv, n)], 1))
        kinds += [kd] * n
    c, kinds = np.concatenate(c), np.array(kinds)
    c, th, local = relax(c, kinds, D, L_tube, l_arm, rng, box, steps=a.steps)

    ss = 3
    Wb, Hb = int(box[0]), int(box[1])
    m = np.zeros((Hb * ss, Wb * ss), np.uint8)
    W = rotate(local, th)
    for i in np.flatnonzero(kinds < 3):
        for sy in (-Hb, 0, Hb):
            off = np.array([0, sy])
            P = W[i] + c[i] + off
            cv2.circle(m, tuple(int(v * ss) for v in P[0]), int(dot_d / 2 * ss), 1, -1, cv2.LINE_AA)
            if kinds[i] == 1:
                cv2.line(m, tuple(int(v * ss) for v in P[0]), tuple(int(v * ss) for v in P[6]), 1, int(line_w * ss), cv2.LINE_AA)
            if kinds[i] == 2:
                for arm in range(3):
                    tip = P[4 + 4 * arm]
                    cv2.line(m, tuple(int(v * ss) for v in P[0]), tuple(int(v * ss) for v in tip), 1, int(line_w * ss), cv2.LINE_AA)
    pits = cv2.resize(m.astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_AREA)
    # IV_V: region nearest to island particles, minus the islands, kept one gap away from pits
    yy, xx = np.mgrid[0:Hb, 0:Wb].astype(np.float32)
    isl = c[kinds == 3]
    near_v = np.full((Hb, Wb), np.inf, np.float32)
    near_o = np.full((Hb, Wb), np.inf, np.float32)
    for pts, arr in ((isl, near_v), (c[kinds < 3], near_o)):
        for px, py in pts:
            for sy in (-Hb, 0, Hb):
                np.minimum(arr, np.hypot(xx - px, yy - py - sy), out=arr)
    pit_bin = (pits > 0.5).astype(np.uint8)
    dist_pit = cv2.distanceTransform(1 - pit_bin, cv2.DIST_L2, 5)
    gap = D - dot_d
    island_r = 9.5  # red island radius whose blurred size equals a blurred Type I pit (15 px)
    purple_v = (near_v < near_o) & (near_v > island_r) & (dist_pit > gap)
    field = np.maximum(pits, purple_v.astype(np.float32))
    phi = cv2.GaussianBlur(field, (0, 0), 4.0)
    ref = np.zeros((61, 61), np.float32)
    cv2.circle(ref, (30, 30), int(round(dot_d / 2)), 1.0, -1, cv2.LINE_AA)
    peak = cv2.GaussianBlur(ref, (0, 0), 4.0).max()
    xq = np.clip((phi / peak - 0.02) / 0.98, 0, 1)
    tq = xq * xq * (3 - 2 * xq)
    rgb = np.clip(RED * (1 - tq[..., None]) + PURPLE * tq[..., None], 0, 255).astype(np.uint8)[:, pad:pad + Wv]
    Image.fromarray(rgb).save(a.out)

    # gap statistics between distinct crypts (skeleton distance - drawn width)
    Wc = W + c[:, None]
    stats = {}
    for kd in range(3):
        sel = np.flatnonzero((kinds == kd) & (c[:, 0] > pad + 20) & (c[:, 0] < pad + Wv - 20))
        others = np.flatnonzero(kinds < 3)
        allp = Wc[others].reshape(-1, 2)
        owner = np.repeat(others, NPTS)
        g = []
        for i in sel:
            dv = Wc[i][:, None] - allp[None]
            dv[..., 1] -= Hb * np.round(dv[..., 1] / Hb)
            d = np.sqrt((dv ** 2).sum(-1))
            d[:, owner == i] = np.inf
            g.append(d.min())
        stats[kd] = (np.median(g), np.percentile(g, 5), np.percentile(g, 95))
    print("skeleton distance to nearest other crypt (target %.1f): " % D +
          ", ".join(f"{n}: median {s[0]:.1f} [5%% {s[1]:.1f}, 95%% {s[2]:.1f}]".replace("%%", "%")
                    for n, s in zip(("I", "IIIL", "IV_B"), stats.values())))

    target = Image.open(ROOT / "docs/images/pit_ideal_sweep.png").convert("RGB")
    sheet = Image.new("RGB", (Wv + 40, 2 * Hv + 150), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((20, 12), "答案見本の候補: pit どうしの間隔（赤い隙間）を一定に保つ", fill="black", font=font(20, True))
    d.text((20, 46), "現在の答案見本", fill="#52514e", font=font(14))
    sheet.paste(target, (20, 66))
    d.text((20, 72 + Hv), "候補: 1 陰窩 = 1 単位、形どうしの隙間が I と同じになるよう位置と向きを緩和（陰窩の間隔は自然に広がる）",
           fill="#52514e", font=font(14))
    sheet.paste(Image.fromarray(rgb), (20, 92 + Hv))
    for cx, nm in zip((0.12, 0.37, 0.62, 0.87), ("Type I", "Type IIIL", "Type IV_B", "Type IV_V")):
        d.text((20 + int(cx * Wv) - 30, 100 + 2 * Hv), nm, fill="black", font=font(16, True))
    sheet.save(a.compare)
    print("saved", a.out, a.compare)


if __name__ == "__main__":
    main()
