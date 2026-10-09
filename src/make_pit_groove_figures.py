#!/usr/bin/env python3
"""Figures for the crest-groove model from saved validation fields.

  sequence.png   representative I / IIIL / IV_B / IV_V panels + crest relief,
                 one validation seed, forks marked in red
  strip.png      every drive of one seed, same crop (continuity of the sweep)
  seeds.png      the four representative drives for several validation seeds
  labels.png     stacked label fractions over the drive axis (all seeds)
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_pit_groove import label
from pit_fork_metrics import fork_overlay

COLORS = {"I": "#2a78d6", "IIIL": "#eb6834", "IV_B": "#1baf7a", "IV_V": "#eda100", "other": "#b9b8b3",
          "empty": "#e3e2de"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fields_for(root: Path, seed: int) -> np.lib.npyio.NpzFile:
    return np.load(root / f"seed{seed}" / "fields.npz")


def panel(w: np.ndarray, crop: int, scale: int, wavelength: float, forks: bool = True) -> Image.Image:
    b = w[:crop, :crop] > 0
    if forks:
        rgb = fork_overlay(w > 0, arm_min=wavelength / 4)[:crop, :crop]
    else:
        rgb = np.full((*b.shape, 3), 238, np.uint8)
        rgb[b] = (70, 58, 120)
    return Image.fromarray(rgb).resize((crop * scale, crop * scale), Image.Resampling.NEAREST)


def relief(v: np.ndarray, crop: int, scale: int) -> Image.Image:
    v = v[:crop, :crop]
    g = ((v - v.min()) / (np.ptp(v) + 1e-12) * 255).astype(np.uint8)
    return Image.fromarray(g).resize((crop * scale, crop * scale), Image.Resampling.BILINEAR)


def grid_image(tiles: list[Image.Image], captions: list[str], title: str, cols: int) -> Image.Image:
    tw, th = tiles[0].size
    pad, cap, head = 10, 34, 44
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (pad + cols * (tw + pad), head + rows * (th + cap + pad)), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((pad, 12), title, fill=INK, font=font(18))
    for i, (t, c) in enumerate(zip(tiles, captions)):
        x = pad + (i % cols) * (tw + pad)
        y = head + (i // cols) * (th + cap + pad)
        img.paste(t, (x, y))
        d.text((x, y + th + 6), c, fill=INK, font=font(14))
    return img


def labels_chart(rows: list[dict], path: Path, title: str) -> None:
    by_s: dict[float, Counter] = {}
    for r in rows:
        by_s.setdefault(float(r["s"]), Counter())[label(r)] += 1
    ss = sorted(by_s)
    W, H, left, right, top, bottom = 1100, 420, 70, 20, 70, 70
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((left, 14), title, fill=INK, font=font(18))
    keys = ["I", "IIIL", "IV_B", "IV_V", "other", "empty"]
    lx = left
    for k in keys[:5]:
        d.rounded_rectangle((lx, 44, lx + 14, 58), 3, fill=COLORS[k])
        d.text((lx + 20, 43), k if k != "other" else "transition", fill=MUTED, font=font(13))
        lx += 110
    pw = (W - left - right) / len(ss)
    ph = H - top - bottom
    for i, s in enumerate(ss):
        n = sum(by_s[s].values())
        y = top + ph
        x0 = left + i * pw + 2
        x1 = left + (i + 1) * pw - 2
        for k in keys:
            c = by_s[s][k]
            if not c:
                continue
            h = ph * c / n
            d.rectangle((x0, y - h + 1, x1, y - 1), fill=COLORS[k])
            y -= h
        d.text((x0, top + ph + 8), f"{s:+.3f}", fill=MUTED, font=font(10))
    d.line((left, top + ph, W - right, top + ph), fill="#c3c2b7")
    for frac in (0.0, 0.5, 0.9, 1.0):
        yy = top + ph * (1 - frac)
        d.text((8, yy - 7), f"{frac:.0%}", fill=MUTED, font=font(11))
        if frac == 0.9:
            d.line((left, yy, W - right, yy), fill="#8a8984")
    d.text((left, H - 30), "invagination drive s  (each bar: independent random initialisations, fraction of seeds per label)",
           fill=MUTED, font=font(12))
    img.save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("artifacts/pit_groove/validation"))
    p.add_argument("--out", type=Path, default=Path("artifacts/pit_groove/figures"))
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    p.add_argument("--reps", type=float, nargs=4, default=[-0.375, -0.10, 0.025, 0.125])
    p.add_argument("--wavelength", type=float, default=24.0)
    p.add_argument("--crop", type=int, default=160)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    names = ["I", "IIIL", "IV_B", "IV_V"]

    f = fields_for(a.root, a.seed)
    tiles = [panel(f[f"seed{a.seed}_s{s:+.4f}_w"], a.crop, 2, a.wavelength) for s in a.reps]
    tiles.append(relief(f[f"seed{a.seed}_s{a.reps[0]:+.4f}_v"], a.crop, 2))
    caps = [f"{n}   s = {s:+.3f}" for n, s in zip(names, a.reps)] + ["crest relief v (fixed, p-independent)"]
    grid_image(tiles, caps, f"Crest-groove model: one equation set, drive s only (seed {a.seed}, red = counted forks)",
               5).save(a.out / "sequence.png")

    keys = sorted({k for k in f.files if k.endswith("_w")}, key=lambda k: float(k.split("_s")[1][:-2]))
    strip = [panel(f[k], 128, 1, a.wavelength, forks=False) for k in keys]
    grid_image(strip, [k.split("_s")[1][:-2] for k in keys], f"Every drive, seed {a.seed} (independent initialisation per panel)",
               11).save(a.out / "strip.png")

    tiles, caps = [], []
    for sd in a.seeds:
        fs = fields_for(a.root, sd)
        for n, s in zip(names, a.reps):
            tiles.append(panel(fs[f"seed{sd}_s{s:+.4f}_w"], 128, 1, a.wavelength))
            caps.append(f"seed {sd}  {n}")
    grid_image(tiles, caps, "Same drives, different random initialisations", 4).save(a.out / "seeds.png")

    rows = []
    for m in glob.glob(str(a.root / "seed*" / "metrics.csv")):
        rows.extend(r for r in csv.DictReader(open(m)) if r["checkpoint"] == "1T")
    labels_chart(rows, a.out / "labels.png", f"Label fractions over {len({r['seed'] for r in rows})} validation seeds")


if __name__ == "__main__":
    main()
