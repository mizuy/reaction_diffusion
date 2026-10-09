#!/usr/bin/env python3
"""Answer-image candidate with constant pit gap AND gradual transitions (drawn target).

Every crypt carries one pit shape that morphs continuously with a progression q(x)
rising smoothly from 0 (left) to 3 (right):
  q 0 -> 1  a dot elongates into a straight tube (two opposite arms grow)
  q 1 -> 2  a third arm sprouts from the middle and the arms open from 180 deg to 120 deg (Y)
  q 2 -> 3  the Y arms reach out and their tips join the neighbours' tips -> network,
            the enclosed red mucosa becomes islands (light/dark inverse of Type I)
Crypts are packed by relaxing position and orientation under a soft repulsion between
skeletons (target distance D = Type I spacing, i.e. a constant red gap).  Tips of two
joining crypts instead attract each other with a strength that grows with q - 2.
Crypt density follows q so that the gap stays the same.  Start: random positions and
orientations.  This is a drawn target, not a simulation.

    uv run python -m pitpattern.figures.make_pit_grade_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pitpattern.figures.make_pit_ideal_sweep import PURPLE, RED, font

ROOT = Path(__file__).resolve().parents[3]
PER_ARM = 4
NPTS = 1 + 3 * PER_ARM


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def shape(q: float, L: float, l: float, l_net: float):
    """(angles, lengths) of the three arms for progression q."""
    if q <= 1:
        s = smooth(q)
        return np.array([0, np.pi, 0.0]), np.array([s * L / 2, s * L / 2, 0.0])
    if q <= 2:
        s = smooth(q - 1)
        open_ = np.pi - s * np.pi / 3                  # 180 -> 120 deg between arms 1 and 2
        ang = np.array([0, open_, -open_ / 2 - (np.pi - open_) * 0 + np.pi])
        ang = np.array([0, open_, 0.5 * (open_ + 2 * np.pi)])  # third arm bisects the reflex side
        lengths = np.array([L / 2 + s * (l - L / 2), L / 2 + s * (l - L / 2), s * l])
        return ang, lengths
    return np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]), np.full(3, l)


def local_points(q, L, l, l_net):
    ang, lens = shape(q, L, l, l_net)
    pts = [np.zeros(2)]
    for a, ln in zip(ang, lens):
        pts += [f * ln * np.array([np.cos(a), np.sin(a)]) for f in np.linspace(1 / PER_ARM, 1, PER_ARM)]
    return np.array(pts)                     # NPTS points; tips at 4, 8, 12


def rotate(local, th):
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    return np.stack([local[..., 0] * c - local[..., 1] * s, local[..., 0] * s + local[..., 1] * c], -1)


def relax(c, q, D, L, l, l_net, rng, box, steps=2000, k=0.25, noise=1.0, max_step=1.0,
          site=None, site_ang=None, k_pin=0.15, dscale=None):
    """site / site_ang: honeycomb site and first-arm angle for IV_V crypts (nan elsewhere)."""
    n = len(c)
    th = rng.uniform(0, 2 * np.pi, n)
    pin = smooth(q - 2) ** 3 if site is not None else np.zeros(n)
    has = ~np.isnan(site[:, 0]) if site is not None else np.zeros(n, bool)
    local = np.stack([local_points(qi, L, l, l_net) for qi in q])
    reach = np.linalg.norm(local, axis=2).max(1)
    ds = np.ones(n) if dscale is None else np.asarray(dscale, float)   # per-crypt spacing factor
    join = pin                                # IV_V tips may meet their neighbours' tips
    tip_mask = np.zeros(NPTS, bool)
    tip_mask[[PER_ARM, 2 * PER_ARM, 3 * PER_ARM]] = True
    inert = np.maximum((local ** 2).sum(axis=(1, 2)), 1.0)
    I = J = None
    for it in range(steps):
        if it % 25 == 0:
            d = c[:, None] - c[None]
            d -= box * np.round(d / box)
            r = np.sqrt((d ** 2).sum(-1))
            I, J = np.nonzero(np.triu(r < D * np.maximum(ds[:, None], ds[None]) + reach[:, None] + reach[None] + 6, 1))
            jp = np.sqrt(join[I] * join[J])          # joining strength of each pair
        W = rotate(local, th)
        sep = c[I] - c[J]
        sep -= box * np.round(sep / box)
        dv = (W[I] + sep[:, None])[:, :, None] - W[J][:, None]
        dist = np.sqrt((dv ** 2).sum(-1)) + 1e-9
        tt = tip_mask[None, :, None] & tip_mask[None, None, :]          # tip-tip pairs
        Dij = D * 0.5 * (ds[I] + ds[J])
        target = np.broadcast_to((Dij * (1 - jp))[:, None, None], dist.shape)   # fusing crypts stop repelling
        over = np.clip(target - dist, 0, None)
        f = (k * over / dist)[..., None] * dv
        fi, fj = f.sum(2), -f.sum(1)
        F, T = np.zeros((n, 2)), np.zeros(n)
        np.add.at(F, I, fi.sum(1))
        np.add.at(F, J, fj.sum(1))
        np.add.at(T, I, (W[I][..., 0] * fi[..., 1] - W[I][..., 1] * fi[..., 0]).sum(1))
        np.add.at(T, J, (W[J][..., 0] * fj[..., 1] - W[J][..., 1] * fj[..., 0]).sum(1))
        if has.any():
            dpin = site[has] - c[has]
            dpin -= box * np.round(dpin / box)
            F[has] += k_pin * pin[has, None] * dpin * np.sqrt(NPTS)
            dth = np.angle(np.exp(3j * (site_ang[has] - th[has]))) / 3      # Y is 120-degree symmetric
            T[has] += 0.0
            th[has] += 0.08 * pin[has] * dth
        amp = noise * max(0.0, 1 - it / (0.7 * steps))
        step = F / np.sqrt(NPTS)
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, max_step / np.maximum(norm, 1e-9))
        c = (c + step + rng.normal(0, amp, c.shape)) % box
        th += np.clip(2.0 * T / inert, -0.05, 0.05) + rng.normal(0, 0.05 * amp, n) * (q > 0.15)
    return c, th, local


def area_per(q, D, L, l, l_net):
    """Area per crypt that keeps the gap D (interpolated between pure shapes)."""
    a0 = D * D * 0.866
    a1 = (D + L) * D * 0.866 / 0.80
    a2 = (D + 1.25 * l) ** 2 * 0.866 / 0.76
    a3 = 2.0 * D * D * 0.866                    # honeycomb: 1 crypt (Y) per 2 islands... see doc
    xs, ys = [0, 1, 2, 3], [a0, a1, a2, a3]
    return np.interp(q, xs, ys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/images/pit_grade_sweep.png")
    ap.add_argument("--compare", type=Path, default=ROOT / "docs/images/pit_grade_sweep_compare.png")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Wv, Hv, pad = 1002, 497, 60
    box = np.array([Wv + 2 * pad, Hv], float)
    D, dot_d, line_w = 27.5, 14.5, 12.5
    L, l, l_net = 20.0, 13.0, 20.0

    def q_of(x):  # x in box coordinates; smooth ramps through the four bands
        u = (x - pad) / Wv
        return np.interp(u, [0.0, 0.09, 0.27, 0.33, 0.48, 0.53, 0.90, 1.0], [0, 0, 1, 1, 2, 2, 3, 3])

    # place crypts by rejection with local density 1 / area_per(q)
    def area(x):
        return area_per(np.minimum(q_of(x), 2), D, L, l, l_net)
    xs = np.linspace(0, box[0], 2000)
    dens = 1 / area(xs)
    n = int(np.trapezoid(dens, xs) * Hv)
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    c = np.stack([np.interp(rng.random(n), cdf, xs), rng.uniform(0, Hv, n)], 1)
    c = c[q_of(c[:, 0]) <= 2]
    # honeycomb template for IV_V: area per site = IV_B area per crypt (crypt number conserved);
    # edge b chosen so that the y period 3b fits the box exactly
    a2 = area_per(2, D, L, l, l_net)
    b = Hv / max(1, round(Hv / (3 * np.sqrt(4 * a2 / (3 * np.sqrt(3)))))) / 3
    l_net = b / 2
    hx = b * np.sqrt(3)
    sites, angs = [], []
    for row in range(int(round(Hv / (1.5 * b)))):
        y0 = row * 1.5 * b
        xoff = (row % 2) * hx / 2
        for kx in range(int(box[0] / hx) + 2):
            x = kx * hx + xoff
            sites += [(x, y0), (x, y0 + b)]
            angs += [np.pi / 2, np.pi / 2 + np.pi]       # A (lower site): arms 90/210/330, B (upper): 270/30/150
    sites, angs = np.array(sites), np.array(angs)
    keep = (q_of(sites[:, 0]) > 2) & (sites[:, 0] < box[0])
    sites, angs = sites[keep], angs[keep]
    # natural imperfection: gentle large-scale warp (periodic in y) plus small jitter
    ph = rng.uniform(0, 2 * np.pi, 4)
    wy = 2 * np.pi / Hv
    sites = sites + np.stack([4.0 * np.sin(sites[:, 1] * wy * 2 + ph[0]) + 2.5 * np.sin(sites[:, 0] / 90 + ph[1]),
                              4.0 * np.sin(sites[:, 0] / 110 + ph[2]) + 2.5 * np.sin(sites[:, 1] * wy + ph[3])], 1)
    sites += rng.normal(0, 1.2, sites.shape)
    jit = 10.0 * (1 - smooth(q_of(sites[:, 0]) - 2))[:, None]
    c = np.concatenate([c, sites % box + rng.normal(0, 1, sites.shape) * jit])
    n = len(c)
    site = np.full((n, 2), np.nan)
    site_ang = np.full(n, np.nan)
    site[n - len(sites):] = sites % box
    site_ang[n - len(sites):] = angs
    q = q_of(c[:, 0])
    c, th, local = relax(c, q, D, L, l, l_net, rng, box, steps=a.steps, site=site, site_ang=site_ang)

    ss = 3
    Wb, Hb = int(box[0]), int(box[1])
    m = np.zeros((Hb * ss, Wb * ss), np.uint8)
    Wpts = rotate(local, th) + c[:, None]
    P = lambda v: (int(v[0] * ss), int(v[1] * ss))
    # Honeycomb crypts: every template edge has a random threshold u; it fuses (groove drawn
    # centre to centre) once the local fusion level smooth(q - 2) exceeds u, so the network
    # grows from Y pairs and small clusters to a full honeycomb.  Arms of unfused edges point
    # at the neighbour and are kept short enough to leave the constant gap D.
    tmpl = np.flatnonzero(~np.isnan(site[:, 0]))
    idx_of = {int(i): k for k, i in enumerate(tmpl)}
    tsites = site[tmpl]
    dd = tsites[:, None] - tsites[None]
    dd[..., 1] -= Hb * np.round(dd[..., 1] / Hb)
    rr = np.sqrt((dd ** 2).sum(-1))
    np.fill_diagonal(rr, np.inf)
    edges = {(int(i), int(j)) for i, j in np.argwhere(rr < 1.2 * l_net * 2) if i < j}
    u = {e: rng.random() for e in edges}
    fused = set()
    for (ki, kj) in edges:
        gi, gj = tmpl[ki], tmpl[kj]
        if smooth(0.5 * (q[gi] + q[gj]) - 2) > u[(ki, kj)]:
            fused.add((ki, kj))
    joins = len(fused)
    nb = {k: [] for k in range(len(tmpl))}
    for (ki, kj) in edges:
        nb[ki].append(kj)
        nb[kj].append(ki)

    def seg(p0, p1):
        for sy in (-Hb, 0, Hb):
            off = np.array([0, sy])
            cv2.line(m, P(p0 + off), P(p1 + off), 1, int(line_w * ss), cv2.LINE_AA)

    for i in range(n):
        for sy in (-Hb, 0, Hb):
            cv2.circle(m, P(c[i] + np.array([0, sy])), int(dot_d / 2 * ss), 1, -1, cv2.LINE_AA)
        if i not in idx_of:
            pts = Wpts[i]
            for arm in range(3):
                tip = pts[(arm + 1) * PER_ARM]
                if np.linalg.norm(tip - pts[0]) > 0.5:
                    seg(pts[0], tip)
            continue
        k = idx_of[i]
        pts = Wpts[i]
        arm_dirs = [pts[(a_ + 1) * PER_ARM] - pts[0] for a_ in range(3)]
        arm_ang = np.array([np.arctan2(v[1], v[0]) for v in arm_dirs])
        replaced = set()
        for kj in nb[k]:
            if (min(k, kj), max(k, kj)) not in fused:
                continue
            j = tmpl[kj]
            d = c[j] - c[i]
            d[1] -= Hb * np.round(d[1] / Hb)
            seg(c[i], c[i] + d / 2)                       # fused groove, centre to centre
            diff = np.abs(np.angle(np.exp(1j * (arm_ang - np.arctan2(d[1], d[0])))))
            replaced.add(int(np.argmin(diff)))           # this arm became the fused groove
        for a_ in range(3):
            if a_ not in replaced and np.linalg.norm(arm_dirs[a_]) > 0.5:
                seg(pts[0], pts[(a_ + 1) * PER_ARM])     # unfused arm keeps its relaxed Y shape
    field = cv2.resize(m.astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_AREA)
    phi = cv2.GaussianBlur(field, (0, 0), 4.0)
    ref = np.zeros((61, 61), np.float32)
    cv2.circle(ref, (30, 30), int(round(dot_d / 2)), 1.0, -1, cv2.LINE_AA)
    peak = cv2.GaussianBlur(ref, (0, 0), 4.0).max()
    xq = np.clip((phi / peak - 0.02) / 0.98, 0, 1)
    tq = xq * xq * (3 - 2 * xq)
    rgb = np.clip(RED * (1 - tq[..., None]) + PURPLE * tq[..., None], 0, 255).astype(np.uint8)[:, pad:pad + Wv]
    Image.fromarray(rgb).save(a.out)

    binary = (phi > 0.5 * peak)[:, pad:pad + Wv]
    red = (~binary).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(red, connectivity=4)
    cx = [st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] / 2 for i in range(1, nl)]
    isl = [(x, 2 * np.sqrt(st[i + 1, cv2.CC_STAT_AREA] / np.pi)) for i, x in enumerate(cx)
           if st[i + 1, cv2.CC_STAT_AREA] < 20000 and st[i + 1, cv2.CC_STAT_LEFT] > 0]
    right = [d for x, d in isl if x > 0.86 * Wv]
    print(f"crypts {n}; joins {joins}; IV_V red islands (right 14%): {len(right)}, equiv. diameter median "
          f"{np.median(right) if right else 0:.1f} px (Type I pit ~15 px)")

    target = Image.open(ROOT / "docs/images/pit_ideal_sweep.png").convert("RGB")
    sheet = Image.new("RGB", (Wv + 40, 2 * Hv + 150), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((20, 12), "答案見本の候補: 間隔一定のまま、形がなめらかに移り変わる", fill="black", font=font(20, True))
    d.text((20, 46), "現在の答案見本", fill="#52514e", font=font(14))
    sheet.paste(target, (20, 66))
    d.text((20, 72 + Hv), "候補: 点が伸びて短管 → 3 本目の腕が生えて Y（ここまで赤い隙間は一定）→ Y どうしが少しずつ融合して蜂の巣状の溝の網、粘膜は大きな島に",
           fill="#52514e", font=font(14))
    sheet.paste(Image.fromarray(rgb), (20, 92 + Hv))
    for cx_, nm in zip((0.06, 0.36, 0.62, 0.92), ("Type I", "Type IIIL", "Type IV_B", "Type IV_V")):
        d.text((20 + int(cx_ * Wv) - 30, 100 + 2 * Hv), nm, fill="black", font=font(16, True))
    sheet.save(a.compare)
    print("saved", a.out, a.compare)


if __name__ == "__main__":
    main()
