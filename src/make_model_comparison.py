#!/usr/bin/env python3
"""One-page comparison of the candidate models tried for the pit-pattern goal.

Each row is one model, each column one value of that model's single control
parameter (low -> high progression).  Every panel starts from independent
random noise (seed 1) with no carried-over state; pits / grooves are dark.
Rows 1-8 are minimal re-implementations of the exploratory runs recorded in
docs/pit_groove_model.md section 4; row 9 is the adopted crest-groove model
read from saved validation fields (seed 100).

    uv run python src/make_model_comparison.py --out docs/images/pit_model_comparison.png
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PIT = np.array([52, 40, 92], np.uint8)
BG = np.array([238, 226, 214], np.uint8)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
VERDICT_COLOR = {"pass": "#1baf7a", "partial": "#eda100", "fail": "#e34948"}


def kgrid(n: int, dx: float):
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return k[:, None] ** 2 + k[None, :] ** 2


def lap(a):
    return np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a


# --- 1/2: phase-field crystal (conserved SH), optional Ohta-Kawasaki term ------------------------
def pfc(psibar, alpha=0.0, r=-0.6, n=128, steps=3000, dt=0.5, seed=1):
    k2 = kgrid(n, np.pi / 4)
    c = 1.0
    den = 1 + dt * (k2 * ((1 - k2) ** 2 + c) + alpha)
    u = psibar + 0.05 * np.random.default_rng(seed).normal(size=(n, n))
    u += psibar - u.mean()
    for _ in range(steps):
        h = (np.fft.fft2(u) - dt * k2 * np.fft.fft2((r - c) * u + u**3)) / den
        h[0, 0] = psibar * n * n
        u = np.fft.ifft2(h).real
    return u > 0


# --- 3: niche-anchored phase separation ------------------------------------------------------------
def niche(m, beta=0.3, lam=18.0, n=128, steps=4000, dt=0.4, seed=1, alpha=0.025):
    k2 = kgrid(n, 1.0)
    kv2 = (2 * np.pi / lam) ** 2
    rng = np.random.default_rng(seed)
    v = 0.05 * rng.normal(size=(n, n))
    u = m + 0.05 * np.random.default_rng(seed + 5000).normal(size=(n, n))
    u += m - u.mean()
    s = 2.0
    denu = 1 + dt * (k2**2 + s * k2 + alpha)
    denv = 1 - dt * (0.2 - (1 - k2 / kv2) ** 2 - 1.0)
    for _ in range(steps):
        v = np.fft.ifft2((np.fft.fft2(v) + dt * np.fft.fft2(v * v - v**3 + v)) / denv).real
        vn = v / (np.abs(v).max() + 1e-9)
        h = (np.fft.fft2(u) - dt * k2 * np.fft.fft2(u**3 - (1 + s) * u - beta * vn)) / denu
        h[0, 0] = m * n * n
        u = np.fft.ifft2(h).real
    return u > 0


# --- 4: asymmetric double-well Ohta-Kawasaki ------------------------------------------------------
def asym_ok(m, a=0.1, alpha=0.006, n=192, steps=6000, dt=0.5, seed=1):
    k2 = kgrid(n, 1.0)
    s = 1.0
    den = 1 + dt * (k2**2 + s * k2 + alpha)
    u = m + 0.05 * np.random.default_rng(seed).normal(size=(n, n))
    u += m - u.mean()
    for _ in range(steps):
        h = (np.fft.fft2(u) - dt * k2 * np.fft.fft2(u * (u - a) * (u - 1) - s * u)) / den
        h[0, 0] = m * n * n
        u = np.fft.ifft2(h).real
    return u > 0.5


# --- 5/6: Gray-Scott, optional long-range inhibitor -----------------------------------------------
def gs_init(n, seed):
    rng = np.random.default_rng(seed)
    s = rng.random((n, n))
    mask = (rng.random((n // 4, n // 4)) < 0.25).repeat(4, 0).repeat(4, 1)
    u = np.ones((n, n))
    v = np.zeros((n, n))
    u[mask] = 0.5
    v[mask] = 0.25
    u += 0.02 * (s - 0.5)
    v += 0.02 * (rng.random((n, n)) - 0.5)
    return u, v


def gray_scott(k, F=0.040, gamma=0.0, n=128, steps=20000, seed=1, ell=15.0, tau=50.0):
    u, v = gs_init(n, seed)
    w = np.zeros((n, n))
    k2 = kgrid(n, 1.0)
    for i in range(steps):
        uvv = u * v * v
        u += 0.2 * lap(u) - uvv + F * (1 - u)
        v += 0.1 * lap(v) + uvv - (F + k + gamma * w) * v
        if gamma and i % 5 == 0:
            w = np.fft.ifft2((np.fft.fft2(w) + 5 * np.fft.fft2(v) / tau) / (1 + 5 * (ell**2 * k2 + 1) / tau)).real
    return v > 0.2


# --- 7: dipolar domains (Garel-Doniach) ------------------------------------------------------------
def dipolar(h, om=2.0, c=0.7, n=128, steps=3000, dt=0.2, seed=1):
    k = 2 * np.pi * np.fft.fftfreq(n)
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    s = 2.0
    den = 1 + dt * (s + om * c - om * kk + kk**2)
    u = 0.1 * np.random.default_rng(seed).normal(size=(n, n))
    for _ in range(steps):
        u = np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(u - u**3 + h + s * u)) / den).real
    return u < 0


# --- 8: quadratic Swift-Hohenberg, pits = low phase -----------------------------------------------
def sh23(r, g=1.0, n=128, steps=3000, dt=0.3, seed=1):
    k2 = kgrid(n, np.pi / 5)
    u = 0.1 * np.random.default_rng(seed).normal(size=(n, n))
    den = 1 - dt * (r - (1 - k2) ** 2 - 1)
    for _ in range(steps):
        u = np.fft.ifft2((np.fft.fft2(u) + dt * np.fft.fft2(g * u * u - u**3 + u)) / den).real
    return u < 0


MODELS = [
    dict(name="1  Phase-field crystal", param="mean density", fn=pfc, kw={},
         values=[-0.30, -0.24, -0.21, -0.18, -0.15, -0.09],
         verdict="fail", note="dots -> stripes abruptly;\nphases separate, stripes align"),
    dict(name="2  PFC + Ohta-Kawasaki", param="mean density", fn=pfc, kw=dict(alpha=0.02),
         values=[-0.27, -0.21, -0.18, -0.15, -0.12, -0.05],
         verdict="fail", note="still abrupt;\nno intermediate types"),
    dict(name="3  Niche-anchored separation", param="pit amount m", fn=niche, kw={},
         values=[-0.6, -0.4, -0.2, -0.1, 0.0, 0.1],
         verdict="fail", note="disks just swell\n(width runaway)"),
    dict(name="4  Asymmetric-well OK", param="pit fraction m", fn=asym_ok, kw={},
         values=[0.22, 0.25, 0.28, 0.30, 0.32, 0.36], crop=160,
         verdict="partial", note="I, IIIL, maze OK; finite\nunits are unbranched worms"),
    dict(name="5  Gray-Scott", param="kill rate k", fn=gray_scott, kw={},
         values=[0.0650, 0.0636, 0.0630, 0.0624, 0.0612, 0.0595],
         verdict="partial", note="dots -> dashes -> maze;\nno branching"),
    dict(name="6  Gray-Scott + inhibitor", param="kill rate k", fn=gray_scott, kw=dict(F=0.062, gamma=0.10, steps=15000),
         values=[0.061, 0.059, 0.056, 0.053, 0.050, 0.047],
         verdict="fail", note="more rods, still\nno branching"),
    dict(name="7  Dipolar (Garel-Doniach)", param="field h", fn=dipolar, kw={},
         values=[0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
         verdict="partial", note="bubbles + rods;\nfew forks"),
    dict(name="8  Quadratic SH (honeycomb)", param="r", fn=sh23, kw={},
         values=[0.02, 0.05, 0.10, 0.20, 0.30, 0.50],
         verdict="partial", note="honeycomb = IV_V template;\nno I->IV_B route"),
]


def run_panel(args):
    i, j = args
    m = MODELS[i]
    return i, j, m["fn"](m["values"][j], **m["kw"])


def groove_row(root: Path, seed: int, drives: list[float]) -> list[np.ndarray]:
    f = np.load(root / f"seed{seed}" / "fields.npz")
    return [f[f"seed{seed}_s{s:+.4f}_w"] > 0 for s in drives]


def font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tile(b: np.ndarray, size: int, crop: int | None) -> Image.Image:
    c = crop or min(b.shape)
    b = b[:c, :c]
    rgb = np.where(b[..., None], PIT, BG).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("docs/images/pit_model_comparison.png"))
    p.add_argument("--groove-root", type=Path, default=Path("artifacts/pit_groove/validation"))
    p.add_argument("--groove-seed", type=int, default=100)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()

    jobs = [(i, j) for i in range(len(MODELS)) for j in range(len(MODELS[i]["values"]))]
    grid: dict[tuple[int, int], np.ndarray] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, j, b in ex.map(run_panel, jobs):
            grid[(i, j)] = b
            print("done", MODELS[i]["name"], MODELS[i]["values"][j], flush=True)

    drives = [-0.375, -0.15, -0.10, 0.025, 0.05, 0.15]
    rows = [dict(name=m["name"], param=m["param"], values=m["values"], verdict=m["verdict"], note=m["note"],
                 crop=m.get("crop"), tiles=[grid[(i, j)] for j in range(len(m["values"]))])
            for i, m in enumerate(MODELS)]
    rows.append(dict(name="9  Crest-groove (adopted)", param="invagination drive s", values=drives, verdict="pass",
                     note="I -> IIIL -> IV_B -> IV_V in order;\n11/12 points >= 18/20 seeds;\nIV_B mostly chains, few Y", crop=200,
                     stages=["I", "IIIL", "IIIL", "IV_B", "IV_B", "IV_V"],
                     tiles=groove_row(a.groove_root, a.groove_seed, drives)))

    ts, gap, left, right, head, rowgap = 150, 6, 250, 260, 92, 34
    W = left + 6 * (ts + gap) + right
    H = head + len(rows) * (ts + rowgap) + 40
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((16, 14), "Candidate models for the pit-pattern goal (I -> IIIL -> IV_B -> IV_V)", fill=INK, font=font(22))
    d.text((16, 46), "Each row: one model, one control parameter increasing to the right. Every panel is an independent "
                     "random start (seed 1); pits/grooves are dark. Panel scales differ between rows.",
           fill=MUTED, font=font(14))
    for i, r in enumerate(rows):
        y = head + i * (ts + rowgap)
        d.text((16, y + 18), r["name"], fill=INK, font=font(17))
        d.text((16, y + 44), r["param"] + "  ->", fill=MUTED, font=font(14))
        for j, (b, val) in enumerate(zip(r["tiles"], r["values"])):
            x = left + j * (ts + gap)
            img.paste(tile(b, ts, r["crop"]), (x, y))
            d.text((x + 2, y + ts + 3), f"{val:g}", fill=MUTED, font=font(12))
            if r.get("stages"):
                d.text((x + ts - 40, y + ts + 2), r["stages"][j], fill=INK, font=font(13))
        x = left + 6 * (ts + gap) + 14
        col = VERDICT_COLOR[r["verdict"]]
        d.rounded_rectangle((x, y + 16, x + 14, y + 30), 3, fill=col)
        d.text((x + 22, y + 14), {"pass": "best (adopted)", "partial": "partial", "fail": "failed"}[r["verdict"]],
               fill=INK, font=font(15))
        d.multiline_text((x, y + 42), r["note"], fill=MUTED, font=font(13), spacing=4)
    d.text((16, H - 28), "Details: docs/pit_groove_model.md (section 4 for rows 1-8, section 3 for row 9).",
           fill=MUTED, font=font(13))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
