#!/usr/bin/env python3
"""Hand-designed target schema for the pit-pattern goal, drawn from the real images.

Convention: dark = pit / groove (where dye pools), light = mucosal surface.
Groove width is the same in every panel; only length, branching and
connectivity change.

  I     small round pits on a near-regular lattice
  IIIL  larger, short tubular / oval pits, separate, random orientation
  IV_B  grooves elongate into short branched, finite trees; mucosa still connected
  IV_V  grooves join into a network; the mucosa becomes winding gyrus-like islands
        of nearly constant width (brain-like), with blind sulci entering them

IV_B and IV_V are taken from one labyrinth field at two levels, so the
B -> V step is continuous in the drawing as well.  This is a drawn target,
not a simulation result.

    uv run python src/pitpattern/figures/make_pit_target_schema.py --out docs/images/pit_target_schema.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pitpattern.metrics.evaluate_pit_steps import prune_skeleton_spurs, zhang_suen_thin

S = 420          # panel size (px)
W = 9            # pit / groove width (px), same in all panels
MUCOSA = np.array([240, 230, 216], np.uint8)
PIT = np.array([47, 58, 107], np.uint8)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
JP = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
JP_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


def font(size, bold=False):
    for n in ((JP_BOLD if bold else JP), JP, "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------- I / IIIL (drawn units)
def type_i(rng) -> np.ndarray:
    m = np.zeros((S, S), np.uint8)
    a = 34.0
    for j in range(-1, int(S / (a * 0.866)) + 2):
        for i in range(-1, int(S / a) + 2):
            x = i * a + (a / 2 if j % 2 else 0) + rng.normal(0, 2.5)
            y = j * a * 0.866 + rng.normal(0, 2.5)
            r = W / 2 + 1 + rng.normal(0, 0.6)
            cv2.circle(m, (int(round(x)), int(round(y))), int(round(r)), 1, -1, cv2.LINE_AA)
    return m > 0


def capsule(m, p0, p1, bend, width):
    t = np.linspace(0, 1, 24)
    mid = (p0 + p1) / 2 + bend
    pts = ((1 - t)[:, None] ** 2 * p0 + 2 * ((1 - t) * t)[:, None] * mid + (t ** 2)[:, None] * p1)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, int(width), cv2.LINE_AA)


def type_iiil(rng) -> np.ndarray:
    m = np.zeros((S, S), np.uint8)
    guard = np.zeros((S, S), np.uint8)
    tries = 0
    while tries < 6000:
        tries += 1
        c = rng.uniform(-10, S + 10, 2)
        L = rng.uniform(1.6, 3.4) * W if rng.random() > 0.15 else rng.uniform(0.2, 0.8) * W
        th = rng.uniform(0, np.pi)
        d = np.array([np.cos(th), np.sin(th)]) * L / 2
        bend = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6) * rng.normal(0, 0.25 * L)
        trial = np.zeros_like(m)
        capsule(trial, c - d, c + d, bend, W + 2 * rng.uniform(0, 1.2))
        if (trial & guard).any():
            continue
        m |= trial
        guard |= cv2.dilate(trial, np.ones((15, 15), np.uint8))
    return m > 0


# ---------------------------------------------------------------- IV (from one labyrinth field)
def labyrinth(rng, wavelength=52.0, band=0.22) -> np.ndarray:
    """Isotropic band-limited random field (no long-range order, unlike a relaxed SH labyrinth)."""
    k = np.fft.fftfreq(S)
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    k0 = 1.0 / wavelength
    filt = np.exp(-((kk - k0) ** 2) / (2 * (band * k0) ** 2))
    u = np.fft.ifft2(np.fft.fft2(rng.normal(size=(S, S))) * filt).real
    return (u - u.mean()) / u.std()


def grooves_from(region: np.ndarray, spur: int) -> np.ndarray:
    sk = prune_skeleton_spurs(zhang_suen_thin(region), min_length=spur)
    g = cv2.dilate(sk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (W, W)))
    return g > 0


def type_iv(u: np.ndarray, frac: float, spur: int) -> np.ndarray:
    """Grooves = midlines of the low region (lowest `frac` of u), drawn at width W."""
    level = np.quantile(u, frac)
    pad = 40  # the field is periodic; pad so thinning has no border artefacts
    up = np.pad(u, pad, mode="wrap")
    g = grooves_from(up < level, spur)
    return g[pad:-pad, pad:-pad]


# ---------------------------------------------------------------- network style (IV_V = inverse of I)
class IslandNetwork:
    """Near-regular mucosal islands; grooves are the island borders, pits sit at triple points.

    I opens only the triple points, IIIL opens isolated single borders, IV_B opens
    finite trees of borders (with Y junctions), IV_V opens every border so the
    mucosa becomes islands -- the light/dark inverse of Type I.
    """

    def __init__(self, rng, spacing=46.0, jitter=2.5):
        pts = []
        rows = int(round(S / (spacing * 0.866)))
        dy = S / rows
        cols = int(round(S / spacing))
        dx = S / cols
        for j in range(rows):
            for i in range(cols):
                pts.append(((i + (0.5 if j % 2 else 0.0)) * dx, (j + 0.5) * dy))
        pts = np.array(pts) + rng.normal(0, jitter, (len(pts), 2))
        self.n = len(pts)
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        best = np.full((3, S, S), np.inf, np.float32)
        lab = np.full((3, S, S), -1, np.int32)
        for k, (px, py) in enumerate(pts):
            for ox in (-S, 0, S):
                for oy in (-S, 0, S):
                    d = np.hypot(xx - (px + ox), yy - (py + oy))
                    # insert d into the sorted top-3 (best[0] <= best[1] <= best[2])
                    c0, c1, c2 = d < best[0], d < best[1], d < best[2]
                    best[2] = np.where(c1, best[1], np.where(c2, d, best[2]))
                    lab[2] = np.where(c1, lab[1], np.where(c2, k, lab[2]))
                    best[1] = np.where(c0, best[0], np.where(c1, d, best[1]))
                    lab[1] = np.where(c0, lab[0], np.where(c1, k, lab[1]))
                    best[0] = np.where(c0, d, best[0])
                    lab[0] = np.where(c0, k, lab[0])
        self.groove = (best[1] - best[0]) < W                    # all borders at width W
        # Triple-point blobs and border pieces are taken per connected component, so the
        # same pair / triple of islands meeting twice (periodic wrap) gives separate items.
        triple = ((best[2] - best[0]) < W * 0.9).astype(np.uint8)
        nv, vlab = cv2.connectedComponents(triple, connectivity=8)
        self.vlab = vlab
        self.vertices = {}
        for v in range(1, nv):
            ys, xs = np.nonzero(vlab == v)
            if len(ys) >= 3:
                self.vertices[v] = (xs.mean(), ys.mean())
        a, b = np.minimum(lab[0], lab[1]), np.maximum(lab[0], lab[1])
        pair = np.where(self.groove, a.astype(np.int64) * self.n + b, -1)
        zone = cv2.dilate(triple, np.ones((3, 3), np.uint8)).astype(bool)
        self.edge_lab = np.zeros((S, S), np.int32)
        self.edge_verts: dict[int, list[int]] = {}
        eid = 0
        for key in np.unique(pair[self.groove & ~zone]):
            comp = ((pair == key) & ~zone).astype(np.uint8)
            ne, elab = cv2.connectedComponents(comp, connectivity=8)
            for c in range(1, ne):
                piece = elab == c
                if piece.sum() < 4:
                    continue
                eid += 1
                self.edge_lab[piece] = eid
                ring = cv2.dilate(piece.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                vs = [int(v) for v in np.unique(vlab[ring]) if v in self.vertices]
                self.edge_verts[eid] = vs
        self.vert_edges: dict[int, list[int]] = {v: [] for v in self.vertices}
        for e, vs in self.edge_verts.items():
            for v in vs:
                self.vert_edges[v].append(e)
        self.edges = [e for e, vs in self.edge_verts.items() if len(vs) == 2]

    def draw(self, open_edges, dots) -> np.ndarray:
        m = np.isin(self.edge_lab, np.array(sorted(open_edges), dtype=np.int32)) if open_edges else np.zeros((S, S), bool)
        touched = {v for e in open_edges for v in self.edge_verts[e]}
        if touched:
            vmask = np.isin(self.vlab, np.array(sorted(touched), dtype=np.int32)).astype(np.uint8)
            # the border pieces were cut 1 px away from the triple-point blobs; seal the seam
            m |= cv2.dilate(vmask, np.ones((5, 5), np.uint8)).astype(bool) & self.groove
        m = m.astype(np.uint8)
        for key in dots:
            x, y = self.vertices[key]
            cv2.circle(m, (int(round(x)), int(round(y))), int(round(W / 2 + 1)), 1, -1, cv2.LINE_AA)
        return m > 0

    def type_i(self):
        return self.draw(set(), list(self.vertices))

    def type_iiil(self, rng, frac=0.45):
        used, opened = set(), set()
        for e in rng.permutation(self.edges):
            e = int(e)
            vs = self.edge_verts[e]
            if any(v in used for v in vs) or rng.random() > frac:
                continue
            opened.add(e)
            used.update(vs)  # tubes never share a triple point, so they stay separate units
        dots = [v for v in self.vertices if v not in used and rng.random() < 0.6]
        return self.draw(opened, dots)

    def type_ivb(self, rng, frac=0.50, max_edges=6):
        parent = {v: v for v in self.vertices}
        size = {v: 0 for v in self.vertices}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        opened = set()
        target = frac * len(self.edges)
        for e in rng.permutation(self.edges):
            e = int(e)
            a, b = (find(v) for v in self.edge_verts[e])
            if a == b or size[a] + size[b] + 1 > max_edges:
                continue  # no loops, finite trees
            parent[a] = b
            size[b] += size[a] + 1
            opened.add(e)
            if len(opened) >= target:
                break
        touched = {v for e in opened for v in self.edge_verts[e]}
        return self.draw(opened, touched | {v for v in self.vertices if rng.random() < 0.3})

    def type_ivv(self):
        return self.draw(set(self.edges), list(self.vertices))


# ---------------------------------------------------------------- renderings
def flat(mask: np.ndarray) -> np.ndarray:
    return np.where(mask[..., None], PIT, MUCOSA).astype(np.uint8)


PASTEL = np.array([[171, 201, 233], [246, 196, 160], [170, 222, 196], [246, 222, 150], [237, 186, 207],
                   [190, 214, 160], [197, 189, 232], [240, 175, 172]], np.uint8)


def mucosa_units(mask: np.ndarray) -> np.ndarray:
    """Colour each connected mucosal region; one colour = still one continuous sheet."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    rgb = np.zeros((S, S, 3), np.uint8)
    big = np.argmax(st[1:, cv2.CC_STAT_AREA]) + 1 if n > 1 else 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 0.6 * S * S:
            rgb[lab == i] = (226, 222, 214)
        else:
            rgb[lab == i] = PASTEL[i % len(PASTEL)]
    rgb[mask] = PIT
    return rgb


def endoscopic(mask: np.ndarray) -> np.ndarray:
    """Rough chromoendoscopy look: raised mucosa, dye pooled in pits."""
    m = mask.astype(np.float32)
    height = cv2.GaussianBlur(1 - m, (0, 0), 4.0)
    gy, gx = np.gradient(height)
    shade = np.clip(0.75 + 6.0 * (-gx * 0.6 - gy * 0.8), 0.45, 1.25)
    dye = cv2.GaussianBlur(m, (0, 0), 1.6)
    base = np.array([214, 160, 140], np.float32)
    blue = np.array([40, 82, 128], np.float32)
    rgb = (base * (1 - dye[..., None]) + blue * dye[..., None]) * shade[..., None]
    spec = np.clip((height - 0.97) * 30, 0, 1) * (shade > 1.05)
    rgb = rgb * (1 - spec[..., None]) + 255 * spec[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def width_px(mask: np.ndarray) -> float:
    sk = zhang_suen_thin(mask)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(np.median(dist[sk.astype(bool)]) * 2) if sk.any() else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("docs/images/pit_target_schema.png"))
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--style", choices=["network", "random"], default="network",
                   help="network: one island lattice, IV_V = inverse of I (default); random: v1 random-field drawing")
    p.add_argument("--b-frac", type=float, default=0.40)
    p.add_argument("--v-frac", type=float, default=0.56)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.style == "network":
        net = IslandNetwork(rng)
        panels = [net.type_i(), net.type_iiil(rng), net.type_ivb(rng), net.type_ivv()]
    else:
        u = labyrinth(rng)
        panels = [type_i(rng), type_iiil(rng), type_iv(u, a.b_frac, spur=6), type_iv(u, a.v_frac, spur=6)]
    heads = ["Type I", "Type IIIL", "Type IV_B", "Type IV_V"]
    if a.style == "network":
        subs = ["粘膜の三重点に小さな丸い pit", "境界が一辺だけ開いた短管。互いに離れる",
                "開いた辺がつながり Y を含む有限の樹。粘膜はまだ一続き", "全境界が開いた網。粘膜が島になる（I の明暗反転）"]
    else:
        subs = ["小さな丸い pit が規則的に並ぶ", "やや大きい短管・楕円の pit。互いに離れる",
                "溝が伸びて短く枝分かれ（有限の樹）。粘膜はまだ一続き", "溝がつながって網に。粘膜は脳回状の島"]
    rows = [("シェーマ", "暗 = pit・溝（色素が溜まる側）\n明 = 粘膜面", flat),
            ("粘膜の連結", "同じ色 = 一続きの粘膜\n色違い = 溝に囲まれた島（脳回）", mucosa_units),
            ("擬似内視鏡像", "実写真と見比べるための\n陰影付き描画（目安）", endoscopic)]

    pad, left, gap, head = 24, 200, 14, 120
    Wimg = left + 4 * (S + gap) + pad
    Himg = head + len(rows) * (S + 40) + 130
    img = Image.new("RGB", (Wimg, Himg), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((pad, 18), "目指す pit pattern のシェーマ（I → IIIL → IV_B → IV_V）", fill=INK, font=font(28, True))
    d.text((pad, 58), "参照シートの実画像をもとに描いた目標像。溝・pit の幅は全段階で同じ（W = 9 px）。"
                      "変わるのは長さ・分岐・連結だけ。シミュレーション結果ではない。", fill=MUTED, font=font(15))
    for j, (h, s) in enumerate(zip(heads, subs)):
        x = left + j * (S + gap)
        d.text((x, head - 40), h, fill=INK, font=font(20, True))
        d.text((x, head - 14), s, fill=MUTED, font=font(13))
    for i, (name, note, fn) in enumerate(rows):
        y = head + 10 + i * (S + 40)
        d.text((pad, y + 6), name, fill=INK, font=font(17, True))
        d.multiline_text((pad, y + 34), note, fill=MUTED, font=font(12), spacing=4)
        for j, m in enumerate(panels):
            img.paste(Image.fromarray(fn(m)), (left + j * (S + gap), y))
    y = head + 10 + len(rows) * (S + 40)
    stats = []
    for h, m in zip(heads, panels):
        n_bg, _, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
        islands = int(sum(1 for i in range(1, n_bg) if st[i, cv2.CC_STAT_AREA] < 0.6 * S * S))
        stats.append(f"{h}: 幅 {width_px(m):.1f} px、被覆 {m.mean():.0%}、粘膜の島 {islands}")
    d.text((pad, y), "実測（この図）: " + " ／ ".join(stats), fill=MUTED, font=font(13))
    if a.style == "network":
        note = ("描き方: ほぼ等間隔に並ぶ粘膜の島（将来の脳回）を置き、その境界を溝、三つの島が接する点を pit の位置とした。"
                "I は三重点だけ、IIIL は孤立した一辺、IV_B は輪を作らない有限の辺の樹、IV_V は全辺。\n"
                "I（明るい粘膜に暗い点）と IV_V（暗い溝の網に明るい島）は明暗が反転した関係になる。"
                "溝の幅は全段階で同じで、変わるのは開いた境界の数とつながり方だけ。")
    else:
        note = ("読み取った特徴: I は暗点が小さく疎（間隔の約 1/3）。IIIL は管が I の点の 2–3 倍の長さで、向きはばらばら。"
                "IV_B は細い溝が短く分岐し、端（盲端）が多い。\n"
                "IV_V は溝が網として閉じ、明るい粘膜が太さほぼ一定のくねった脳回になる（脳回の幅は溝の幅の数倍）。"
                "平行ストライプでも、太く膨らんだ塊でもない。")
    d.multiline_text((pad, y + 28), note, fill=MUTED, font=font(13), spacing=6)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out)
    print("saved", a.out, img.size)
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
