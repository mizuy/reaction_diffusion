#!/usr/bin/env python3
"""Answer-image candidate: Round -> Elongated -> Branch -> Gyrus -> Villous (drawn target).

Progression q(x) rises smoothly from 0 (left) to 4 (right).  One crypt = one pit unit;
the red gap between pit grooves stays the Type I gap up to Branch and widens from Gyrus on.
  q 0-1  Round -> Elongated   a dot grows into a straight tube
  q 1-2  Elongated -> Branch  a third arm sprouts, the tube opens into a Y (one fork)
  q 2-3  Branch -> Gyrus      the Y arms keep growing tip-wise, turning and occasionally
                              forking; a Swift-Hohenberg relaxation then joins them into long,
                              smooth, winding grooves whose pitch widens slowly (D -> 1.25 D)
  q 3-4  Gyrus -> Villous     within the same field the pitch widens to the honeycomb spacing
                              and a quadratic term breaks the red ridges into elongated
                              islands, which round up and lock onto the honeycomb (villi)
Crypt positions/orientations for q <= 2 come from the constant-gap relaxation of
make_pit_grade_sweep; growth is a self-avoiding tip growth with a length budget that
increases with q.  This is a drawn target, not a simulation of tissue.

    uv run python -m pitpattern.figures.make_pit_gyrus_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pitpattern.figures.make_pit_grade_sweep import PER_ARM, area_per, relax, rotate, smooth
from pitpattern.figures.make_pit_ideal_sweep import PURPLE, RED, font

ROOT = Path(__file__).resolve().parents[3]


class Hash:
    """Spatial hash of centreline points (periodic in y) with owner and arc position."""

    def __init__(self, cell, Hb):
        self.cell, self.Hb, self.d = cell, Hb, {}

    def add(self, p, owner, arc, branch=-1):
        key = (int(p[0] // self.cell), int((p[1] % self.Hb) // self.cell))
        self.d.setdefault(key, []).append((p[0], p[1] % self.Hb, owner, arc, branch))

    def nearest_foreign(self, p, owner, arc, skip, radius, core=0.0, anc=None):
        """Distance to the closest point that is not (same owner and within `skip` along arc)."""
        cx, cy = int(p[0] // self.cell), int((p[1] % self.Hb) // self.cell)
        r = int(np.ceil(radius / self.cell))
        ny = int(np.ceil(self.Hb / self.cell))
        best = np.inf
        y = p[1] % self.Hb
        for gx in range(cx - r, cx + r + 1):
            for gy in range(cy - r, cy + r + 1):
                for qx, qy, o, a, b in self.d.get((gx, gy % ny), ()):
                    # own groove: ignore the stretch just behind the tip (same branch or the
                    # ancestor stretch before the fork) and, right at the start, the crypt core
                    if o == owner:
                        if a < core:
                            continue
                        if anc is not None and b in anc and a <= anc[b] and abs(a - arc) < skip:
                            continue
                    dy = qy - y
                    dy -= self.Hb * round(dy / self.Hb)
                    dd = np.hypot(qx - p[0], dy)
                    if dd < best:
                        best = dd
        return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/images/pit_gyrus_sweep.png")
    ap.add_argument("--compare", type=Path, default=ROOT / "docs/images/pit_gyrus_sweep_compare.png")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--extra", type=float, default=420.0, help="extra groove length per crypt at full gyrus (px)")
    ap.add_argument("--sparse", type=float, default=3.0, help="relative increase of area per crypt in gyrus/villous")
    ap.add_argument("--hole", type=float, default=1.15, help="refill red areas farther than hole*D from grooves")
    ap.add_argument("--refill-from", type=float, default=0.6, help="refill where smooth(q-2) exceeds this")
    ap.add_argument("--cut", type=float, default=0.0, help="spacing of ridge cuts in villous (units of D); 0 = off")
    ap.add_argument("--refill", action="store_true", help="refill gyrus holes with extra branches")
    ap.add_argument("--tube-len", type=float, default=36.0, help="final IIIL tube length (px)")
    ap.add_argument("--arm-len", type=float, default=24.0, help="IV_B arm length (px)")
    ap.add_argument("--villous-area", type=float, default=0.35, help="area per crypt in villous / Branch")
    ap.add_argument("--fork", type=float, default=0.12, help="fork probability per step in gyrus")
    ap.add_argument("--maze-steps", type=int, default=180, help="Swift-Hohenberg steps for the gyrus maze (0 = off)")
    ap.add_argument("--blend", choices=["continuous", "patchy"], default="continuous")
    ap.add_argument("--m-from", type=float, default=2.2)
    ap.add_argument("--m-gain", type=float, default=3.0)
    ap.add_argument("--h-from", type=float, default=3.25)
    ap.add_argument("--h-gain", type=float, default=1.6)
    ap.add_argument("--pin", type=float, default=0.25)
    ap.add_argument("--g-villous", type=float, default=-1.5, help="quadratic term at full villous (< 0: red islands)")
    ap.add_argument("--lam-from", type=float, default=2.7, help="q where the groove pitch starts to widen")
    ap.add_argument("--lam-gyrus", type=float, default=1.25, help="groove pitch at full gyrus (units of D)")
    ap.add_argument("--lam-villous", type=float, default=0.0, help="groove pitch at villous (units of D); 0 = honeycomb row spacing")
    ap.add_argument("--q-knots", type=float, nargs=4, default=[0.57, 0.73, 0.92, 1.0],
                    help="u where q reaches 3, leaves 3, reaches 4, ends")
    ap.add_argument("--no-honeycomb", action="store_true", help="never paste the drawn honeycomb; villous comes from the field")
    ap.add_argument("--g-from", type=float, default=2.95)
    ap.add_argument("--g-gain", type=float, default=1.2)
    ap.add_argument("--seg-short", type=float, default=1.0, help="maze segment length at gyrus onset (units of D)")
    ap.add_argument("--seg-long", type=float, default=6.0, help="maze segment length just before cuts stop (units of D)")
    ap.add_argument("--cut-until", type=float, default=2.9, help="no cuts beyond this q (long maze into villous)")
    ap.add_argument("--maze-g", type=float, default=0.0, help="quadratic term (favours junctions)")
    ap.add_argument("--maze-r", type=float, default=0.3, help="Swift-Hohenberg control parameter")
    ap.add_argument("--maze-cover", type=float, default=0.0, help="purple fraction of the maze (groove width / spacing)")
    ap.add_argument("--spread", type=float, default=1.0, help="how much of the surface expansion goes into crypt spacing")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Wv, Hv, pad = 1002, 497, 60
    box = np.array([Wv + 2 * pad, Hv], float)
    D, dot_d, line_w = 27.5, 14.5, 12.5
    L, l = a.tube_len, a.arm_len
    Wb, Hb = int(box[0]), int(box[1])

    def q_of(x):
        u = (x - pad) / Wv
        return np.interp(u, [0.0, 0.03, 0.20, 0.23, 0.31, 0.43, *a.q_knots],
                         [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def expand(qq):
        return 1 + a.sparse * smooth((np.asarray(qq) - 2) / 2)

    # crypts: constant-gap packing with density of the shape at min(q, 2)
    def area(x):
        qq = q_of(x)
        # protruding lesion: crypt number is conserved while the surface expands, so the
        # number of crypts per unit area falls continuously from Branch to Villous
        return area_per(np.minimum(qq, 2), D, L, l, l) * expand(qq)
    xs = np.linspace(0, box[0], 2000)
    dens = 1 / area(xs)
    n = int(np.trapezoid(dens, xs) * Hv)
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    c = np.stack([np.interp(rng.random(n), cdf, xs), rng.uniform(0, Hv, n)], 1)
    c = c[q_of(c[:, 0]) <= 3]
    # Villous (q 3 -> 4): crypts settle onto a honeycomb with the same crypt density
    # (as in make_pit_grade_sweep) and neighbouring crypts fuse edge by edge.
    a2 = area_per(2, D, L, l, l) * a.villous_area     # honeycomb: area per crypt in villous
    b = Hv / max(1, round(Hv / (3 * np.sqrt(4 * a2 / (3 * np.sqrt(3)))))) / 3
    hx = b * np.sqrt(3)
    sites, angs = [], []
    for row in range(int(round(Hv / (1.5 * b)))):
        y0 = row * 1.5 * b
        xoff = (row % 2) * hx / 2
        for kx in range(int(box[0] / hx) + 2):
            x = kx * hx + xoff
            sites += [(x, y0), (x, y0 + b)]
            angs += [np.pi / 2, np.pi / 2 + np.pi]
    sites, angs = np.array(sites), np.array(angs)
    keep = (q_of(sites[:, 0]) > 3) & (sites[:, 0] < box[0])
    sites, angs = sites[keep], angs[keep]
    ph = rng.uniform(0, 2 * np.pi, 4)
    wy = 2 * np.pi / Hv
    sites = sites + np.stack([4.0 * np.sin(sites[:, 1] * wy * 2 + ph[0]) + 2.5 * np.sin(sites[:, 0] / 90 + ph[1]),
                              4.0 * np.sin(sites[:, 0] / 110 + ph[2]) + 2.5 * np.sin(sites[:, 1] * wy + ph[3])], 1)
    sites += rng.normal(0, 1.2, sites.shape)
    c = np.concatenate([c, sites % box + rng.normal(0, 10, sites.shape) * (1 - smooth(q_of(sites[:, 0]) - 3))[:, None]])
    n = len(c)
    site = np.full((n, 2), np.nan)
    site_ang = np.full(n, np.nan)
    site[n - len(sites):] = sites % box
    site_ang[n - len(sites):] = angs
    q = q_of(c[:, 0])
    q_r = np.where(q > 3, q - 1, np.minimum(q, 2))      # relax(): shape Y, pin/fusion from q_r - 2
    # surface expansion: crypts keep a wider spacing where the surface has grown (Branch -> Villous)
    dscale = np.where(q > 3, 1.0, 1 + a.spread * (np.sqrt(expand(np.minimum(q, 3))) - 1))
    c, th, local = relax(c, q_r, D, L, l, l, rng, box, steps=a.steps, site=site, site_ang=site_ang, dscale=dscale)
    Wpts = rotate(local, th) + c[:, None]

    # honeycomb edges and which of them have fused
    tmpl = np.flatnonzero(~np.isnan(site[:, 0]))
    ts = site[tmpl]
    dd = ts[:, None] - ts[None]
    dd[..., 1] -= Hb * np.round(dd[..., 1] / Hb)
    rr = np.sqrt((dd ** 2).sum(-1))
    np.fill_diagonal(rr, np.inf)
    fused_pairs = []
    for ki, kj in np.argwhere(rr < 1.25 * b):
        if ki < kj:
            gi, gj = tmpl[ki], tmpl[kj]
            if smooth(0.5 * (q[gi] + q[gj]) - 3) > rng.random():
                fused_pairs.append((gi, gj))

    # ---------------------------------------------------------------- tip growth (q > 2)
    H = Hash(D, Hb)
    segs = []
    tips = []
    replaced = np.zeros((n, 3), bool)
    arm_ang = np.array([[np.arctan2(*(Wpts[i, (k + 1) * PER_ARM] - c[i])[::-1]) for k in range(3)] for i in range(n)])
    for gi, gj in fused_pairs:
        d = c[gj] - c[gi]
        d[1] -= Hb * np.round(d[1] / Hb)
        for i, v in ((gi, d), (gj, -d)):
            k = int(np.argmin(np.abs(np.angle(np.exp(1j * (arm_ang[i] - np.arctan2(v[1], v[0])))))))
            replaced[i, k] = True
            segs.append((c[i].copy(), c[i] + v / 2))
            for f in np.linspace(0.2, 1, 5):
                H.add(c[i] + f * v / 2, i, f * np.linalg.norm(v) / 2, 3 * i + k)
    for i in range(n):
        H.add(c[i], i, 0.0)
        for arm in range(3):
            if replaced[i, arm]:
                continue
            tip = Wpts[i, (arm + 1) * PER_ARM]
            v = tip - c[i]
            ln = np.linalg.norm(v)
            if ln < 0.5:
                continue
            segs.append((c[i], tip))
            bid = 3 * i + arm
            for f in np.linspace(0.25, 1, 4):
                H.add(c[i] + f * v, i, f * ln, bid)
            g = smooth(1.4 * (q[i] - 2)) * (1 - smooth(q[i] - 3))   # gyrus growth fades out into villous
            if g > 0.02:
                sib = {3 * i + k: l + D / np.sqrt(3) for k in range(3)}
                sib[bid] = np.inf
                tips.append([i, tip.copy(), np.arctan2(v[1], v[0]), ln, g * a.extra / 3, bid, sib])
    step, D_c = 2.0, D            # centreline spacing = D keeps the red gap of Type I
    stats = {"blocked": 0, "budget": 0, "refill": 0}
    state = {"bid": 3 * n}

    def grow(active):
        while active:
            nxt = []
            for t in rng.permutation(len(active)):
                o, p, h, arc, bud, bid, anc = active[t]
                if bud <= 0:
                    stats["budget"] += 1
                    continue
                g = smooth(q[o] - 2)
                moved = False
                for dh in sorted([0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.25, -1.25],
                                 key=lambda x: abs(x) + 0.2 * rng.random()):
                    hh = h + dh + rng.normal(0, 0.08)
                    np_ = p + step * np.array([np.cos(hh), np.sin(hh)])
                    if np_[0] < 0 or np_[0] > Wb:
                        continue
                    core = l + 1 if arc < l + D_c else 0.0
                    if H.nearest_foreign(np_, o, arc + step, skip=1.05 * D_c, radius=D_c, core=core, anc=anc) >= D_c:
                        segs.append((p.copy(), np_.copy()))
                        H.add(np_, o, arc + step, bid)
                        nxt.append([o, np_, hh, arc + step, bud - step, bid, anc])
                        if rng.random() < (a.fork + 0.10 * (abs(dh) > 0.8)) * g:
                            side = hh + rng.choice([-1, 1]) * np.pi / 3
                            near = arc + step + 1.3 * D_c
                            nb_ = state["bid"]
                            state["bid"] += 1
                            nanc = dict(anc)
                            nanc[bid] = near
                            nanc[nb_] = np.inf
                            anc[nb_] = near
                            nxt.append([o, np_.copy(), side, arc + step, (bud - step) * 0.6, nb_, nanc])
                        moved = True
                        break
                if not moved:
                    stats["blocked"] += 1
            active = nxt

    grow(tips)

    # ---------------------------------------------------------------- refill gyrus holes
    # Red areas farther than `hole` from every groove centreline (in the gyrus/villous zone)
    # get a new branch from the nearest groove, aimed into the hole.
    hole = a.hole * D_c
    failed = np.zeros((Hb, Wb), bool)
    xs_ = np.arange(Wb)[None, :].repeat(Hb, 0).astype(float)
    zone = smooth(q_of(xs_) - 2) > a.refill_from
    for it in range(600 if a.refill else 0):
        cl = np.zeros((Hb, Wb), np.uint8)
        for s0, s1 in segs:
            for sy in (-Hb, 0, Hb):
                cv2.line(cl, (int(s0[0]), int(s0[1] + sy)), (int(s1[0]), int(s1[1] + sy)), 1, 1)
        for i in range(n):
            cv2.circle(cl, (int(c[i, 0]), int(c[i, 1])), 1, 1, -1)
        dist = cv2.distanceTransform(1 - cl, cv2.DIST_L2, 5)
        cand = np.where(zone & ~failed, dist, 0)
        y0, x0 = np.unravel_index(np.argmax(cand), cand.shape)
        if cand[y0, x0] < hole:
            break
        hp = np.array([x0, y0], float)
        # nearest stored groove point (spatial hash search)
        best, root = np.inf, None
        cx_, cy_ = int(hp[0] // H.cell), int(hp[1] // H.cell)
        ny = int(np.ceil(Hb / H.cell))
        for gx in range(cx_ - 3, cx_ + 4):
            for gy in range(cy_ - 3, cy_ + 4):
                for qx, qy, o, arc_, b_ in H.d.get((gx, gy % ny), ()):
                    dy = qy - hp[1]
                    dy -= Hb * round(dy / Hb)
                    dd = np.hypot(qx - hp[0], dy)
                    if dd < best:
                        best, root = dd, (np.array([qx, hp[1] + dy]), o, arc_, b_)
        n_before = len(segs)
        if root is not None:
            rp, owner, oarc, ob = root
            nb_ = state["bid"]
            state["bid"] += 1
            anc = {nb_: np.inf, ob: oarc + 1.3 * D_c}
            for o2, b2 in ((owner, b) for b in range(3 * owner, 3 * owner + 3)):
                anc.setdefault(b2, l + D_c)     # own core arms near the centre
            h0 = np.arctan2(hp[1] - rp[1], hp[0] - rp[0])
            grow([[owner, rp.copy(), h0, oarc, a.extra, nb_, anc]])
        if len(segs) - n_before < 3:            # could not grow: do not try this hole again
            yy_, xx_ = np.ogrid[:Hb, :Wb]
            failed |= (xx_ - x0) ** 2 + (yy_ - y0) ** 2 < (0.8 * D_c) ** 2
        else:
            stats["refill"] += 1

    # ---------------------------------------------------------------- villous: cut red ridges
    # Medial axis of the red ridges in the villous zone; cut points every `cut` px along the
    # ridges, kept with probability smooth(q - 3); each cut is a short groove across the ridge.
    from pitpattern.metrics.evaluate_pit_steps import zhang_suen_thin
    pur = np.zeros((Hb, Wb), np.uint8)
    for s0, s1 in segs:
        for sy in (-Hb, 0, Hb):
            cv2.line(pur, (int(s0[0]), int(s0[1] + sy)), (int(s1[0]), int(s1[1] + sy)), 1, int(line_w))
    for i in range(n):
        cv2.circle(pur, (int(c[i, 0]), int(c[i, 1])), int(dot_d / 2), 1, -1)
    ridge = zhang_suen_thin((pur == 0)).astype(bool)
    dist_r, lab_r = cv2.distanceTransformWithLabels(1 - pur, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zero_pur = np.argwhere(pur > 0)
    pts = np.argwhere(ridge)
    rng.shuffle(pts)
    chosen = []
    bridges = 0
    cut = a.cut * D_c
    for y, x in (pts if a.cut > 0 else []):
        qq = q_of(float(x))
        if qq <= 3 or dist_r[y, x] > 0.8 * D_c:
            continue
        if any((x - cx) ** 2 + (y - cy) ** 2 < cut * cut for cx, cy in chosen):
            continue
        chosen.append((x, y))
        if rng.random() >= smooth(1.6 * (qq - 3)):
            continue
        # width direction: the line through the ridge point with the shortest purple-to-purple span
        best = None
        for ang in np.linspace(0, np.pi, 12, endpoint=False):
            u = np.array([np.cos(ang), np.sin(ang)])
            ends_ = []
            for sgn in (1, -1):
                for r_ in range(1, int(1.2 * D_c)):
                    xx_, yy_ = int(round(x + sgn * r_ * u[0])), int(round(y + sgn * r_ * u[1])) % Hb
                    if not (0 <= xx_ < Wb):
                        break
                    if pur[yy_, xx_]:
                        ends_.append(r_)
                        break
            if len(ends_) == 2 and (best is None or sum(ends_) < best[0]):
                best = (sum(ends_), u, ends_)
        if best is None:
            continue
        _, u, (r1, r2) = best
        Pm = np.array([x, y], float)
        A, B = Pm + r1 * u, Pm - r2 * u
        segs.append((A, B))
        bridges += 1

    # ---------------------------------------------------------------- render
    ss = 3
    m = np.zeros((Hb * ss, Wb * ss), np.uint8)
    P = lambda v: (int(v[0] * ss), int(v[1] * ss))
    for sy in (-Hb, 0, Hb):
        off = np.array([0, sy])
        for i in range(n):
            cv2.circle(m, P(c[i] + off), int(dot_d / 2 * ss), 1, -1, cv2.LINE_AA)
        for s0, s1 in segs:
            cv2.line(m, P(s0 + off), P(s1 + off), 1, int(line_w * ss), cv2.LINE_AA)
    field = cv2.resize(m.astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_AREA)

    # Gyrus: start from the grown trees and let a Swift-Hohenberg relaxation with stripe period D
    # equalise the groove spacing (fills the wide gaps, pushes close grooves apart) -> dense maze
    # with constant width/spacing that inherits the branching of the trees.  Blended in by w(q).
    if a.maze_steps > 0:
        k = 2 * np.pi * np.fft.fftfreq(Wb)
        ky = 2 * np.pi * np.fft.fftfreq(Hb)
        K2 = ky[:, None] ** 2 + k[None, :] ** 2
        k0 = 2 * np.pi / D
        lin = a.maze_r - (1 - K2 / k0 ** 2) ** 2
        dt = 0.3
        den = 1 - dt * (lin - 1)
        qx = q_of(np.arange(Wb, dtype=float))
        # groove pitch lam(x): D up to the gyrus onset, lam_gyrus at full gyrus, the honeycomb
        # row spacing at villous.  The step is a blend of three constant-pitch steps.
        lam_v = a.lam_villous * D if a.lam_villous > 0 else 1.5 * b
        levels = np.array([D, max(a.lam_gyrus * D, D + 1e-3), max(lam_v, a.lam_gyrus * D + 2e-3)])
        lam_x = np.interp(qx, [a.lam_from, 3.0, 4.0], levels)
        dens = [1 - dt * (a.maze_r - (1 - K2 / (2 * np.pi / L_) ** 2) ** 2 - 1) for L_ in levels]
        wts = [np.interp(lam_x, levels, np.eye(3)[i])[None, :] for i in range(3)]
        if a.blend == "continuous":
            # one field for Branch -> Gyrus -> Villous: the relaxation rate m(x) rises from 0
            # (trees kept) to 1 (equal-spacing maze); towards Villous a pinning h(x) pulls the
            # same field onto the honeycomb network drawn there.
            m_x = smooth(a.m_gain * (qx - a.m_from))[None, :]
            h_x = smooth(a.h_gain * (qx - a.h_from))[None, :]
            T = 2 * cv2.GaussianBlur(field, (0, 0), 1.5) - 1
            u0 = 2 * cv2.GaussianBlur(field, (0, 0), 2.0) - 1
            u0 = u0 - m_x * (1 - h_x) * cv2.GaussianBlur(u0, (0, 0), D)
            u = u0 + 0.15 * m_x * (1 - h_x) * rng.normal(size=u0.shape)
            # optional: a quadratic term growing towards Villous turns the maze into a network
            # of red islands within the same field (labyrinth -> hexagonal islands)
            g_x = a.g_villous * smooth(a.g_gain * (qx - a.g_from))[None, :]
            for _ in range(a.maze_steps):
                F = np.fft.fft2(u) + dt * np.fft.fft2(u - u ** 3 + (a.maze_g + g_x) * u * u)
                u_sh = sum(w_ * np.fft.ifft2(F / d_).real for w_, d_ in zip(wts, dens))
                u = u + np.maximum(m_x, h_x) * (u_sh - u) + a.pin * h_x * (T - u)
        else:
            u0 = 2 * cv2.GaussianBlur(field, (0, 0), 2.0) - 1
            u0 -= cv2.GaussianBlur(u0, (0, 0), D)        # remove the local mean: stripes nucleate in empty red too
            u = 0.8 * u0 / max(np.abs(u0).max(), 1e-6) + 0.15 * rng.normal(size=u0.shape)
            for _ in range(a.maze_steps):
                u = np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(u - u ** 3 + a.maze_g * u * u)) / den).real
        thr = np.quantile(u, 1 - a.maze_cover) if a.maze_cover > 0 else 0.0   # 0: zero level
        maze = (u > thr).astype(np.float32)
        maze = cv2.erode(maze, np.ones((2, 2), np.uint8)) if a.maze_cover > 0.5 else maze
        # constant groove width: keep only the centreline of the maze and redraw it with the
        # same line width as every other stage (thresholding alone gives beaded grooves)
        from pitpattern.metrics.evaluate_pit_steps import prune_skeleton_spurs, zhang_suen_thin
        mb = maze > 0.5
        nlab, lab_, st_, _ = cv2.connectedComponentsWithStats(mb.astype(np.uint8), connectivity=8)
        mb = np.isin(lab_, 1 + np.flatnonzero(st_[1:, cv2.CC_STAT_AREA] >= 0.6 * line_w ** 2))
        pad_y = 8   # periodic in y: pad so thinning has no border artefacts
        mb_p = np.concatenate([mb[-pad_y:], mb, mb[:pad_y]])
        sk = prune_skeleton_spurs(zhang_suen_thin(mb_p), min_length=4)[pad_y:-pad_y]
        # short -> long maze: cut the centreline at points whose spacing grows with q, leaving a
        # red gap of the Type I width (gap + line width) at every cut
        qx_ = q_of(np.arange(Wb, dtype=float))
        seg_len = np.where((qx_ < a.cut_until) & (qx_ > 2.3), np.maximum(np.interp(qx_, [2.4, a.cut_until], [a.seg_short, a.seg_long]), 0.5) * D, np.inf)
        _unused = np.where(qx_ < a.cut_until, np.interp(qx_, [2.4, a.cut_until], [a.seg_short, a.seg_long]) * D, np.inf)
        r_cut = 0.5 * ((D - dot_d) + line_w)
        pts_ = np.argwhere(sk)
        rng.shuffle(pts_)
        cuts = []
        for y_, x_ in pts_:
            L_ = seg_len[x_]
            if not np.isfinite(L_):
                continue
            # planar Poisson-disk radius sqrt(L*D) gives a mean groove length ~L between cuts
            if all((x_ - cx) ** 2 + min(abs(y_ - cy), Hb - abs(y_ - cy)) ** 2 >= L_ * D for cy, cx in cuts):
                cuts.append((y_, x_))
        if cuts:
            yy_, xx_ = np.mgrid[0:Hb, 0:Wb]
            for cy, cx in cuts:
                dy_ = np.minimum(abs(yy_ - cy), Hb - abs(yy_ - cy))
                sk &= (xx_ - cx) ** 2 + dy_ ** 2 > r_cut ** 2
        ys_, xs_s = np.nonzero(sk)
        mm = np.zeros((Hb * ss, Wb * ss), np.uint8)
        for y_, x_ in zip(ys_, xs_s):
            for sy in (-Hb, 0, Hb):
                cv2.circle(mm, (int((x_ + 0.5) * ss), int((y_ + 0.5 + sy) * ss)), int(line_w / 2 * ss), 1, -1)
        maze = cv2.resize(mm.astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_AREA)
        if a.blend == "continuous":
            act = np.maximum(m_x, h_x)
            keep = (act < 0.02) | ((h_x > 0.98) & (not a.no_honeycomb))   # untouched trees / pure honeycomb
            field = np.where(np.broadcast_to(keep, field.shape), field, maze)
            qx = None
        wx = None if qx is None else smooth(2.0 * (qx - 2.5)) * (1 - smooth(1.6 * (qx - 3.05)))
    if a.maze_steps > 0 and wx is not None:
        # crisp patchy switch (no semi-transparent ghosting): each region takes one pattern
        noise = cv2.GaussianBlur(rng.random((Hb, Wb)).astype(np.float32), (0, 0), D * 0.6)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        use_maze = wx[None, :] > noise
        # holes of the tree pattern inside the transition are filled with the maze as well
        tree_bin = (field > 0.5).astype(np.uint8)
        dist_tree = cv2.distanceTransform(1 - tree_bin, cv2.DIST_L2, 5)
        hole = ((dist_tree > 0.8 * D) & (wx[None, :] > 0.02)).astype(np.uint8)
        hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.6 * D),) * 2)).astype(bool)
        hole &= dist_tree > 0.45 * D          # keep one gap away from the tree grooves
        use_maze |= hole
        field = np.where(use_maze, maze, field)
    phi = cv2.GaussianBlur(field, (0, 0), 4.0)
    ref = np.zeros((61, 61), np.float32)
    cv2.circle(ref, (30, 30), int(round(dot_d / 2)), 1.0, -1, cv2.LINE_AA)
    peak = cv2.GaussianBlur(ref, (0, 0), 4.0).max()
    xq = np.clip((phi / peak - 0.02) / 0.98, 0, 1)
    tq = xq * xq * (3 - 2 * xq)
    rgb = np.clip(RED * (1 - tq[..., None]) + PURPLE * tq[..., None], 0, 255).astype(np.uint8)[:, pad:pad + Wv]
    Image.fromarray(rgb).save(a.out)
    print(f"crypts {n}, segments {len(segs)}, tips {len(tips)}, blocked {stats['blocked']}, out of budget {stats['budget']}, "
          f"refill branches {stats['refill']}, ridge cuts {bridges}")

    target = Image.open(ROOT / "docs/images/pit_ideal_sweep.png").convert("RGB")
    sheet = Image.new("RGB", (Wv + 40, 2 * Hv + 150), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((20, 12), "答案見本の候補: Round → Elongated → Branch → Gyrus → Villous", fill="black", font=font(20, True))
    d.text((20, 46), "現在の答案見本", fill="#52514e", font=font(14))
    sheet.paste(target, (20, 66))
    d.text((20, 72 + Hv), "候補: 点 → 短管 → Y → 腕が伸びてつながり長い脳回状の溝に → 畝が島に切れて絨毛状に（隙間は Gyrus から徐々に広がる）",
           fill="#52514e", font=font(14))
    sheet.paste(Image.fromarray(rgb), (20, 92 + Hv))
    for cx_, nm in zip((0.03, 0.22, 0.40, 0.63, 0.90), ("Round (I)", "Elongated (IIIL)", "Branch (IV_B)", "Gyrus", "Villous (IV_V)")):
        d.text((20 + int(cx_ * Wv), 100 + 2 * Hv), nm, fill="black", font=font(15, True))
    sheet.save(a.compare)
    print("saved", a.out, a.compare)


if __name__ == "__main__":
    main()
