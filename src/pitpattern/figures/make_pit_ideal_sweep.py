#!/usr/bin/env python3
"""Corrected ideal sweep: the slide-6 reaction-diffusion look without runaway elongation.

goal_rd_sweep_i_to_iv.png has the desired pit width, spacing and regular
arrangement, but in its middle the pits keep elongating into long stripes.
This drawing keeps the same lattice and widths (measured from that image)
and lets the units stop at a finite size:

  I     purple dots on a triangular lattice (the image's right end)
  IIIL  two neighbouring dots joined through one triangle centre -> short tube
  IV_B  three dots joined at one triangle centre -> Y; a few Ys joined -> small trees
  IV_V  every triangle centre of one orientation joined -> purple honeycomb with
        red dots left over, on the same spacing as the I dots (the image's left end)

All joins go through the same set of triangle centres, so every stage is a
subset of the IV_V network and the purple width is the same everywhere.
This is a drawn target, not a simulation.

    uv run python src/pitpattern/figures/make_pit_ideal_sweep.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
PURPLE = np.array([81, 10, 109], np.float32)  # extreme colours sampled from slide 6
RED = np.array([216, 10, 3], np.float32)
JP = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
JP_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


def font(size, bold=False):
    for n in ((JP_BOLD if bold else JP), JP, "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


class Lattice:
    def __init__(self, rng, w, h, a, warp=5.0, jitter=1.0, rotate=11.0):
        self.a = a
        dy = a * np.sqrt(3) / 2
        nr, nc = int(1.5 * max(w, h) / dy), int(1.5 * max(w, h) / a)
        idx, pos = {}, []
        for r in range(-nr // 3, nr):
            for c in range(-nc // 3, nc):
                idx[(r, c)] = len(pos)
                pos.append(((c + 0.5 * (r % 2)) * a, r * dy))
        pos = np.array(pos, float)
        th = np.deg2rad(rotate)
        cx, cy = w / 2, h / 2
        x0, y0 = pos[:, 0] - cx, pos[:, 1] - cy
        pos = np.stack([cx + x0 * np.cos(th) - y0 * np.sin(th), cy + x0 * np.sin(th) + y0 * np.cos(th)], 1)
        # gentle large-scale distortion like a real (slightly imperfect) hexagonal pattern
        ph = rng.uniform(0, 2 * np.pi, 4)
        pos[:, 0] += warp * np.sin(pos[:, 1] / 90 + ph[0]) + warp * 0.6 * np.sin(pos[:, 0] / 130 + ph[1])
        pos[:, 1] += warp * np.sin(pos[:, 0] / 110 + ph[2]) + warp * 0.6 * np.sin(pos[:, 1] / 70 + ph[3])
        pos += rng.normal(0, jitter, pos.shape)
        self.pos = pos
        # down-pointing triangles (r,c),(r,c+1),(r+1,·): their centres are the join points
        # IV_V joins through the down-pointing triangles only; IV_B Ys may use either
        self.tris, self.down = [], []
        for (r, c), i in idx.items():
            j = idx.get((r, c + 1))
            for dr, down in ((1, True), (-1, False)):
                k = idx.get((r + dr, c + (r % 2)))
                if j is not None and k is not None:
                    self.tris.append((i, j, k))
                    self.down.append(down)
        self.down = np.array(self.down)
        self.centres = np.array([pos[list(t)].mean(0) for t in self.tris])
        self.nbrs = [[] for _ in pos]
        for (r, c), i in idx.items():
            for dr, dc in ((0, 1), (1, r % 2), (1, r % 2 - 1)):
                j = idx.get((r + dr, c + dc))
                if j is not None:
                    self.nbrs[i].append(j)
                    self.nbrs[j].append(i)
        self.vtris = [[] for _ in pos]
        for t, tri in enumerate(self.tris):
            for v in tri:
                self.vtris[v].append(t)


def build(rng, lat: Lattice, width, height, zones, border=14.0):
    """Assign every dot to a stage by position, then cover each stage's dots completely.

    Inside a band every dot takes part in that stage's unit (no leftover Type I dots);
    only near the band borders (position noise `border` px) do neighbouring stages mix.
    """
    x = lat.pos[:, 0] + rng.normal(0, border, len(lat.pos))
    stage = np.rint(np.interp(x / width, zones["x"], zones["p"])).astype(int)  # 0=I 1=IIIL 2=IV_B 3=IV_V
    free = np.ones(len(lat.pos), bool)
    joins = []

    def take(t=None, arms=()):
        joins.append((t, list(arms)))
        free[list(arms)] = False

    # IV_V: every down-triangle whose dots are (mostly) in the IV_V band joins
    for t, tri in enumerate(lat.tris):
        if lat.down[t] and sum(stage[v] == 3 for v in tri) >= 2:
            take(t, tri)

    # IV_B: cover with Ys, choosing the most constrained dot first so few are left over
    def y_options(v):
        return [t for t in lat.vtris[v] if all(free[u] and stage[u] == 2 for u in lat.tris[t])]

    pool = [v for v in rng.permutation(len(lat.pos)) if stage[v] == 2]
    while True:
        cand = [v for v in pool if free[v]]
        if not cand:
            break
        opts = {v: y_options(v) for v in cand}
        cand = [v for v in cand if opts[v]]
        if not cand:
            break
        v = min(cand, key=lambda u: (len(opts[u]), rng.random()))
        t = opts[v][rng.integers(len(opts[v]))]
        take(t, lat.tris[t])
    owner = {}
    for idx, (t, arms) in enumerate(joins):
        for u in arms:
            owner[u] = idx
    tree = {idx: idx for idx in owner.values()}

    def root(a):
        while tree[a] != a:
            a = tree[a]
        return a

    size = {}
    for v in [v for v in pool if free[v]]:  # leftovers become a branch of a neighbouring Y
        hosts = [u for u in lat.nbrs[v] if u in owner and stage[u] == 2]
        if hosts:
            u = hosts[rng.integers(len(hosts))]
            take(None, (v, u))
            owner[v] = owner[u]
    # join a few neighbouring Ys into small trees (finite, no loops)
    for v in rng.permutation(pool):
        if v not in owner or rng.random() > 0.07:
            continue
        for u in rng.permutation(lat.nbrs[v]):
            if u in owner and stage[u] == 2 and owner[u] != owner[v]:
                a, b = root(owner[u]), root(owner[v])
                if a == b or size.get(a, 1) + size.get(b, 1) > 2:
                    continue
                take(None, (v, u))
                tree[a] = b
                size[b] = size.get(a, 1) + size.get(b, 1)
                break

    # IIIL: pair dots into short straight tubes, most constrained first (near-perfect matching)
    pool = [v for v in rng.permutation(len(lat.pos)) if stage[v] == 1]
    while True:
        cand = [v for v in pool if free[v]]
        opts = {v: [u for u in lat.nbrs[v] if free[u] and stage[u] == 1] for v in cand}
        cand = [v for v in cand if opts[v]]
        if not cand:
            break
        v = min(cand, key=lambda u: (len(opts[u]), rng.random()))
        u = opts[v][rng.integers(len(opts[v]))]
        take(None, (v, u))
    return joins, stage


def neighbours(lat, v):
    d = np.hypot(*(lat.pos - lat.pos[v]).T)
    return np.flatnonzero((d > 1) & (d < lat.a * 1.2))


def render(lat, joins, w, h, dot_d, line_w, blur=4.0, ss=3):
    m = np.zeros((h * ss, w * ss), np.uint8)
    for x, y in lat.pos:
        cv2.circle(m, (int(x * ss), int(y * ss)), int(dot_d / 2 * ss), 1, -1, cv2.LINE_AA)
    for t, arms in joins:
        if t is None:  # straight tube between two neighbouring dots
            (x0, y0), (x1, y1) = lat.pos[arms[0]], lat.pos[arms[1]]
            cv2.line(m, (int(x0 * ss), int(y0 * ss)), (int(x1 * ss), int(y1 * ss)), 1, int(line_w * ss), cv2.LINE_AA)
            continue
        cx, cy = lat.centres[t]
        for v in arms:
            x, y = lat.pos[v]
            cv2.line(m, (int(cx * ss), int(cy * ss)), (int(x * ss), int(y * ss)), 1, int(line_w * ss), cv2.LINE_AA)
        cv2.circle(m, (int(cx * ss), int(cy * ss)), int(line_w / 2 * ss), 1, -1, cv2.LINE_AA)
    # The slide-6 spots are soft bumps (purple core, gradual fall to red), so blur the
    # drawn shape and map its intensity straight to colour, normalised by an isolated dot.
    small = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    phi = cv2.GaussianBlur(small, (0, 0), blur)
    ref = np.zeros((61, 61), np.float32)
    cv2.circle(ref, (30, 30), int(round(dot_d / 2)), 1.0, -1, cv2.LINE_AA)
    peak = cv2.GaussianBlur(ref, (0, 0), blur).max()
    x = np.clip((phi / peak - 0.02) / 0.98, 0, 1)
    t = x * x * (3 - 2 * x)  # smoothstep: red stays red between pits, cores reach purple
    rgb = RED * (1 - t[..., None]) + PURPLE * t[..., None]
    binary = phi > 0.5 * peak
    return np.clip(rgb, 0, 255).astype(np.uint8), binary


def build_funnel(rng, lat: Lattice, width, zones, border=14.0, tip_tube=12.0, tip_y=11.5):
    """Per-crypt units (one unit per dot; density unchanged), as in the funnel picture.

    Each crypt mouth stays centred on its dot and grows into the gaps of the lattice
    (towards triangle centres = corners of its hexagonal territory):
      IIIL: a short tube along one gap axis (tips towards one up- and one down-corner),
      IV_B: a three-armed Y with arms towards three corners of one orientation,
      IV_V: arms reach the down-corners and join the neighbours' arms (network).
    Axes / orientations are chosen most-constrained-first so that few corners receive
    two tips.  Returns a list of segments (x0, y0, x1, y1) to draw on top of the dots.
    """
    x = lat.pos[:, 0] + rng.normal(0, border, len(lat.pos))
    stage = np.rint(np.interp(x / width, zones["x"], zones["p"])).astype(int)
    use = np.zeros(len(lat.tris), int)          # how many tips point into each corner
    segs = []
    up_of = [[t for t in lat.vtris[v] if not lat.down[t]] for v in range(len(lat.pos))]
    dn_of = [[t for t in lat.vtris[v] if lat.down[t]] for v in range(len(lat.pos))]

    def tip(v, t, length):
        c = lat.centres[t] - lat.pos[v]
        return lat.pos[v] + c / np.linalg.norm(c) * length

    def axes(v):  # gap axes: a down-corner and the up-corner opposite to it
        out = []
        for td in dn_of[v]:
            cd = lat.centres[td] - lat.pos[v]
            best = min(up_of[v], key=lambda tu: np.dot(cd, lat.centres[tu] - lat.pos[v]), default=None)
            if best is not None:
                out.append((td, best))
        return out

    order = [v for v in rng.permutation(len(lat.pos))]
    # IV_V: arms to all three down-corners, long enough to meet the neighbours there
    for v in order:
        if stage[v] == 3:
            for t in dn_of[v]:
                segs.append((*lat.pos[v], *lat.centres[t]))
    # IIIL tubes: choose the axis whose two corners are least used
    for v in sorted([v for v in order if stage[v] == 1], key=lambda u: rng.random()):
        ax = axes(v)
        if not ax:
            continue
        td, tu = min(ax, key=lambda a: (use[a[0]] + use[a[1]], rng.random()))
        use[td] += 1
        use[tu] += 1
        segs.append((*tip(v, td, tip_tube - 5.75), *tip(v, tu, tip_tube - 5.75)))
    # IV_B Ys: choose the orientation (down or up corners) with the least used corners
    for v in [v for v in order if stage[v] == 2]:
        opts = [o for o in (dn_of[v], up_of[v]) if len(o) == 3]
        if not opts:
            continue
        ts = min(opts, key=lambda o: (sum(use[t] for t in o), rng.random()))
        for t in ts:
            use[t] += 1
            segs.append((*lat.pos[v], *tip(v, t, tip_y - 5.75)))
    return segs, stage


def render_segments(lat, segs, w, h, dot_d, line_w, blur=4.0, ss=3):
    m = np.zeros((h * ss, w * ss), np.uint8)
    for x, y in lat.pos:
        cv2.circle(m, (int(x * ss), int(y * ss)), int(dot_d / 2 * ss), 1, -1, cv2.LINE_AA)
    for x0, y0, x1, y1 in segs:
        cv2.line(m, (int(x0 * ss), int(y0 * ss)), (int(x1 * ss), int(y1 * ss)), 1, int(line_w * ss), cv2.LINE_AA)
    small = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    phi = cv2.GaussianBlur(small, (0, 0), blur)
    ref = np.zeros((61, 61), np.float32)
    cv2.circle(ref, (30, 30), int(round(dot_d / 2)), 1.0, -1, cv2.LINE_AA)
    peak = cv2.GaussianBlur(ref, (0, 0), blur).max()
    xq = np.clip((phi / peak - 0.02) / 0.98, 0, 1)
    tq = xq * xq * (3 - 2 * xq)
    rgb = RED * (1 - tq[..., None]) + PURPLE * tq[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8), phi > 0.5 * peak


class ExpandingLattice:
    """Crypt centres whose spacing a(x) grows with progression (surface expansion of a
    protruding lesion): soft-repulsion relaxation from random positions with a
    position-dependent range, so dislocations appear where the spacing changes.
    Triangles come from the Delaunay triangulation."""

    def __init__(self, rng, w, h, a_of_x, steps=900, margin=60):
        self.w, self.h = w, h
        xs = np.linspace(-margin, w + margin, 400)
        dens = 2 / (np.sqrt(3) * a_of_x(xs) ** 2)
        n = int(np.trapezoid(dens, xs) * (h + 2 * margin))
        cdf = np.cumsum(dens)
        cdf /= cdf[-1]
        x = np.stack([np.interp(rng.random(n), cdf, xs), rng.uniform(-margin, h + margin, n)], 1)
        lo, hi = np.array([-margin, -margin]), np.array([w + margin, h + margin])
        for it in range(steps):
            d = x[:, None, :] - x[None, :, :]
            r = np.sqrt((d ** 2).sum(-1)) + np.eye(n) * 1e9
            a_loc = a_of_x(x[:, 0])
            rng_ij = 1.15 * 0.5 * (a_loc[:, None] + a_loc[None, :])
            f = np.clip(rng_ij - r, 0, None) / r
            x += 0.25 * (d * f[..., None]).sum(1) + rng.normal(0, 1.5 * (1 - it / steps), x.shape)
            x = np.clip(x, lo, hi)
        self.pos = x
        self.a_loc = a_of_x(x[:, 0])
        sub = cv2.Subdiv2D((int(lo[0]) - 5, int(lo[1]) - 5, int(hi[0] - lo[0]) + 10, int(hi[1] - lo[1]) + 10))
        for px, py in x:
            sub.insert((float(px), float(py)))
        key = {(round(px, 3), round(py, 3)): i for i, (px, py) in enumerate(x)}
        tris = []
        for t in sub.getTriangleList():
            ids = [key.get((round(float(t[2 * k]), 3), round(float(t[2 * k + 1]), 3))) for k in range(3)]
            if None in ids:
                continue
            P = x[ids]
            e = [np.linalg.norm(P[0] - P[1]), np.linalg.norm(P[1] - P[2]), np.linalg.norm(P[0] - P[2])]
            if max(e) > 1.5 * self.a_loc[ids].mean():
                continue  # hull slivers
            tris.append(tuple(ids))
        self.tris = tris
        self.centres = np.array([x[list(t)].mean(0) for t in tris])
        self.vtris = [[] for _ in x]
        for ti, t in enumerate(tris):
            for v in t:
                self.vtris[v].append(ti)
        edge_tris = {}
        for ti, t in enumerate(tris):
            for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
                edge_tris.setdefault(tuple(sorted(e)), []).append(ti)
        self.tri_adj = [set() for _ in tris]
        for ts in edge_tris.values():
            for a_ in ts:
                self.tri_adj[a_].update(b for b in ts if b != a_)


def build_expanding(rng, lat: ExpandingLattice, width, zones, border=14.0, f_tube=0.62, f_y=0.42):
    """One unit per crypt: IIIL tube along a gap axis, IV_B Y to alternate corners,
    IV_V arms to an independent set of corners shared with the neighbours (network)."""
    x = lat.pos[:, 0] + rng.normal(0, border, len(lat.pos))
    stage = np.rint(np.interp(x / width, zones["x"], zones["p"])).astype(int)
    segs, use = [], np.zeros(len(lat.tris), int)

    def corners(v):  # this crypt's corners sorted by angle
        ts = lat.vtris[v]
        ang = [np.arctan2(*(lat.centres[t] - lat.pos[v])[::-1]) for t in ts]
        return [t for _, t in sorted(zip(ang, ts))]

    def tip(v, t, f):
        return lat.pos[v] + f * (lat.centres[t] - lat.pos[v])

    # IV_V: an independent set of corners (no two chosen corners share an edge), grown by
    # BFS through "second neighbours" (corners sharing only a vertex, on the other side of
    # it) so that one sublattice is picked over large domains -> regular islands
    chosen = np.zeros(len(lat.tris), bool)
    in_v = np.array([sum(stage[v] == 3 for v in t) >= 2 for t in lat.tris])
    second = [set() for _ in lat.tris]   # same-sublattice corners: two steps around a shared vertex
    for v in range(len(lat.pos)):
        ts = lat.vtris[v]
        if len(ts) < 4:
            continue
        ang = [np.arctan2(*(lat.centres[t] - lat.pos[v])[::-1]) for t in ts]
        ring = [t for _, t in sorted(zip(ang, ts))]
        m = len(ring)
        for i, t in enumerate(ring):
            second[t].update((ring[(i + 2) % m], ring[(i - 2) % m]))
    seen = np.zeros(len(lat.tris), bool)
    for start in rng.permutation(np.flatnonzero(in_v)):
        if seen[start]:
            continue
        queue = [start]
        seen[start] = True
        while queue:
            t = queue.pop(0)
            if not any(chosen[u] for u in lat.tri_adj[t]):
                chosen[t] = True
                for u in second[t]:
                    if in_v[u] and not seen[u]:
                        seen[u] = True
                        queue.append(u)
    for t in np.flatnonzero(chosen):
        for v in lat.tris[t]:
            if stage[v] == 3:
                segs.append((*lat.pos[v], *lat.centres[t]))
    order = list(rng.permutation(len(lat.pos)))
    # IV_B: three arms towards alternate neighbours (edge directions).  Each edge may carry
    # one arm only; there are 3 edges per crypt, so alternate triples can tile without contact.
    nbrs = [set() for _ in lat.pos]
    for t in lat.tris:
        for i in t:
            nbrs[i].update(j for j in t if j != i)
    claimed = set()
    todo = [v for v in order if stage[v] == 2]
    while todo:
        best = None
        for v in todo:
            ns = sorted(nbrs[v], key=lambda j: np.arctan2(*(lat.pos[j] - lat.pos[v])[::-1]))
            if len(ns) < 5:
                continue
            for k in (0, 1):
                trip = ns[k::2][:3]
                cost = sum((min(v, j), max(v, j)) in claimed for j in trip)
                key = (cost, rng.random())
                if best is None or key < best[0]:
                    best = (key, v, trip)
        if best is None:
            break
        _, v, trip = best
        todo.remove(v)
        for j in trip:
            claimed.add((min(v, j), max(v, j)))
            segs.append((*lat.pos[v], *(lat.pos[v] + f_y * (lat.pos[j] - lat.pos[v]))))
    for v in [v for v in order if stage[v] == 1]:
        cs = corners(v)
        m = len(cs)
        if m < 4:
            continue
        axes = [(cs[k], cs[(k + m // 2) % m]) for k in range(m)]
        t1, t2 = min(axes, key=lambda a: (use[a[0]] + use[a[1]], rng.random()))
        use[t1] += 1
        use[t2] += 1
        segs.append((*tip(v, t1, f_tube), *tip(v, t2, f_tube)))
    return segs, stage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "docs/images/pit_ideal_sweep.png")
    p.add_argument("--compare", type=Path, default=ROOT / "docs/images/pit_ideal_sweep_compare.png")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--style", choices=["pairs", "funnel", "expand"], default="pairs",
                   help="pairs: tubes/Ys join 2/3 dots (density halves); funnel: one tube / Y per dot (density kept)")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    W, H = 1002, 497
    spacing, dot_d, line_w = 27.5, 14.5, 11.5  # matched to goal_rd_sweep_i_to_iv.png (NN 27.4 px)
    lat = Lattice(rng, W, H, spacing)
    zones = dict(x=[0.0, 0.24, 0.26, 0.49, 0.51, 0.74, 0.76, 1.0], p=[0, 0.49, 1, 1.49, 2, 2.49, 3, 3])
    if a.style == "expand":
        a_of_x = lambda xx: np.interp(np.asarray(xx) / W, [0.0, 0.18, 0.34, 0.55, 1.0],
                                      [spacing, spacing, 1.45 * spacing, 1.6 * spacing, 1.6 * spacing])
        lat = ExpandingLattice(rng, W, H, a_of_x)
        segs, stage = build_expanding(rng, lat, W, zones)
        rgb, binary = render_segments(lat, segs, W, H, dot_d, 12.5)
        Image.fromarray(rgb).save(a.out)
        print("saved", a.out, "crypts per band (I, IIIL, IV_B, IV_V):",
              [int(((stage == k) & (lat.pos[:, 0] > 0) & (lat.pos[:, 0] < W)).sum()) for k in range(4)])
        return
    elif a.style == "funnel":
        segs, stage = build_funnel(rng, lat, W, zones)
        rgb, binary = render_segments(lat, segs, W, H, dot_d, line_w)
        joins = []
    else:
        joins, stage = build(rng, lat, W, H, zones)
        rgb, binary = render(lat, joins, W, H, dot_d, line_w)
    Image.fromarray(rgb).save(a.out)

    # comparison: original (mirrored so that Type I is on the left) above the corrected drawing
    orig = Image.open(ROOT / "docs/images/goal_rd_sweep_i_to_iv.png").convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    orig = orig.resize((W, H))
    pad, head, lab = 20, 60, 40
    sheet = Image.new("RGB", (W + 2 * pad, head + 2 * (H + lab) + 120), "#fcfcfb")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 14), "修正した理想像: 太さ・間隔は slide 6 のまま、管や分岐は伸び続けずに止まる", fill="#0b0b0b", font=font(22, True))
    y = head
    for title, im in (("元画像（goal_rd_sweep_i_to_iv.png を左右反転して I を左に）", orig), ("修正版（描画・シミュレーションではない）", Image.fromarray(rgb))):
        d.text((pad, y), title, fill="#52514e", font=font(15))
        sheet.paste(im, (pad, y + 24))
        y += H + lab
    centers = [0.09, 0.35, 0.62, 0.90]
    for c, name in zip(centers, ["Type I", "Type IIIL", "Type IV_B", "Type IV_V"]):
        d.text((pad + int(c * W) - 30, y - 10), name, fill="#0b0b0b", font=font(17, True))
    d.multiline_text((pad, y + 20),
                     "I: 赤地に紫の点（元画像の右端と同じ）。IIIL: 隣の点と1か所だけつながったまっすぐな短管で止まる。"
                     "IV_B: 3点が三角形の中心でつながった Y と、その小さな樹で止まる。\n"
                     "IV_V: 全部の中心がつながり紫の網になり、赤い点が同じ間隔で残る＝ I の明暗反転（元画像の左端と同じ）。"
                     f"紫の太さ {line_w:.0f}–{dot_d:.0f} px・点の間隔 {spacing:.0f} px は元画像の実測値。",
                     fill="#52514e", font=font(13), spacing=6)
    sheet.save(a.compare)
    n, _, st, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    used = np.zeros(len(lat.pos), bool)
    for t, arms in joins:
        used[arms] = True
    inside = (lat.pos[:, 0] > 0) & (lat.pos[:, 0] < W) & (lat.pos[:, 1] > 0) & (lat.pos[:, 1] < H)
    for k, name in ((1, "IIIL"), (2, "IV_B")):
        sel = inside & (stage == k)
        print(f"{name}: dots left unjoined {int((~used & sel).sum())} / {int(sel.sum())}")
    print("saved", a.out, a.compare, "purple fraction by fifth:",
          [round(float(binary[:, i * W // 5:(i + 1) * W // 5].mean()), 2) for i in range(5)])


if __name__ == "__main__":
    main()
