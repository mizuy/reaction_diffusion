#!/usr/bin/env python3
"""Figures for the crypt lobulation model.

  pit_crypt_sweep_compare.png  answer image (top) vs one model field with p rising left to right (bottom)
  pit_crypt_seeds.png          the four bands for several validation seeds (independent random starts)
  pit_crypt_labels.png         label counts over the validation seeds at every registered p

    uv run python -m pitpattern.figures.make_pit_crypt_figures
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from PIL import ImageFont

from pitpattern.figures.make_pit_groove_figures import COLORS, INK, MUTED, SURFACE

ROOT = Path(__file__).resolve().parents[3]


def font(size: int, bold: bool = False):
    for name in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sweep_compare(sweep: Path, out: Path) -> None:
    target = Image.open(ROOT / "docs/images/pit_ideal_sweep.png").convert("RGB")
    model = Image.open(sweep).convert("RGB").resize(target.size)
    pad, head, gap = 20, 56, 34
    W = target.width + 2 * pad
    sheet = Image.new("RGB", (W, head + 2 * (target.height + gap) + 70), SURFACE)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 14), "Crypt lobulation model vs answer image", fill=INK, font=font(22, True))
    y = head
    for title, im in (("Answer image (drawn target, docs/images/pit_ideal_sweep.png)", target),
                      ("Crypt lobulation model: one field, p rising linearly from -0.5 (left) to 3.4 (right)", model)):
        d.text((pad, y), title, fill=MUTED, font=font(14))
        sheet.paste(im, (pad, y + 20))
        y += im.height + gap
    for c, name in zip((0.10, 0.36, 0.62, 0.88), ("Type I", "Type IIIL", "Type IV_B", "Type IV_V")):
        d.text((pad + int(c * target.width) - 30, y - 8), name, fill=INK, font=font(16, True))
    sheet.save(out)


def seeds_sheet(root: Path, out: Path, seeds, ps) -> None:
    tiles = [[Image.open(root / f"seed{s}" / f"seed{s}_p{p:.2f}.png").convert("RGB") for p in ps] for s in seeds]
    tw, th = tiles[0][0].size
    pad, head, cap = 10, 52, 22
    sheet = Image.new("RGB", (pad + len(ps) * (tw + pad), head + len(seeds) * (th + cap)), SURFACE)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), "Same p, independent random starts (validation seeds)", fill=INK, font=font(18, True))
    names = ["I", "IIIL", "IV_B", "IV_V"]
    for i, row in enumerate(tiles):
        for j, im in enumerate(row):
            x, y = pad + j * (tw + pad), head + i * (th + cap)
            sheet.paste(im, (x, y))
            d.text((x + 2, y + th + 3), f"seed {seeds[i]}   {names[j]}  p={ps[j]:.1f}", fill=MUTED, font=font(12))
    sheet.save(out)


def labels_chart(root: Path, out: Path) -> None:
    rows = [r for f in glob.glob(str(root / "seed*" / "metrics.csv")) for r in csv.DictReader(open(f))]
    by_p: dict[float, Counter] = {}
    for r in rows:
        by_p.setdefault(float(r["p"]), Counter())[r["label"]] += 1
    ps = sorted(by_p)
    W, H, left, top, bottom = 900, 360, 60, 70, 60
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((left, 14), f"Crypt model: labels over {len({r['seed'] for r in rows})} validation seeds", fill=INK, font=font(18, True))
    lx = left
    for k in ("I", "IIIL", "IV_B", "IV_V", "other"):
        d.rounded_rectangle((lx, 44, lx + 14, 58), 3, fill=COLORS[k])
        d.text((lx + 20, 43), k if k != "other" else "transition", fill=MUTED, font=font(13))
        lx += 110
    pw, ph = (W - left - 20) / len(ps), H - top - bottom
    for i, p in enumerate(ps):
        n = sum(by_p[p].values())
        y = top + ph
        for k in ("I", "IIIL", "IV_B", "IV_V", "transition"):
            c = by_p[p][k]
            if c:
                h = ph * c / n
                d.rectangle((left + i * pw + 2, y - h + 1, left + (i + 1) * pw - 2, y - 1),
                            fill=COLORS["other" if k == "transition" else k])
                y -= h
        d.text((left + i * pw + 4, top + ph + 6), f"{p:.1f}", fill=MUTED, font=font(11))
    d.text((left, H - 26), "progression p (registered evaluation points)", fill=MUTED, font=font(12))
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "artifacts/pit_crypt/validation")
    ap.add_argument("--sweep", type=Path, default=ROOT / "artifacts/pit_crypt/sweep/sweep_seed1.png")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    a = ap.parse_args()
    sweep_compare(a.sweep, a.out / "pit_crypt_sweep_compare.png")
    seeds_sheet(a.root, a.out / "pit_crypt_seeds.png", a.seeds, [0.2, 0.9, 2.3, 3.2])
    labels_chart(a.root, a.out / "pit_crypt_labels.png")
    print("saved figures to", a.out)


if __name__ == "__main__":
    main()
