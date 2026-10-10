#!/usr/bin/env python3
"""Figures for docs/models/pit_mouth_model.md from artifacts/pit_mouth (run the make targets first).

    uv run python -m pitpattern.figures.make_pit_mouth_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from pitpattern.figures.make_pit_ideal_sweep import font

ROOT = Path(__file__).resolve().parents[3]
INK, MUTED, BG = "black", "#52514e", "#fcfcfb"
STAGES = ("Round", "Elongated", "Branch", "Gyrus", "Villous")


def sweep_compare(sweep: Path, out: Path) -> None:
    target = Image.open(ROOT / "docs/images/pit_gyrus_sweep.png").convert("RGB")
    model = Image.open(sweep).convert("RGB")
    pad = 20
    sheet = Image.new("RGB", (target.width + 2 * pad, target.height + model.height + 150), BG)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), "陰窩口モデルと答案見本", fill=INK, font=font(20, True))
    d.text((pad, 46), "答案見本（描画）", fill=MUTED, font=font(14))
    sheet.paste(target, (pad, 66))
    y = 72 + target.height
    d.text((pad, y), "陰窩口モデル: 1 枚の場で p を左 0 から右 4 へ変えた計算（乱数初期化）", fill=MUTED, font=font(14))
    sheet.paste(model, (pad, y + 20))
    for cx, nm in zip((0.03, 0.22, 0.40, 0.63, 0.90), ("Round (I)", "Elongated (IIIL)", "Branch (IV_B)", "Gyrus", "Villous (IV_V)")):
        d.text((pad + int(cx * target.width), y + 28 + model.height), nm, fill=INK, font=font(15, True))
    sheet.save(out)


def seeds_sheet(root: Path, out: Path, seeds, ps, tile: int = 220) -> None:
    pad, lab_w = 14, 150
    sheet = Image.new("RGB", (lab_w + len(seeds) * (tile + 6) + pad, 60 + len(ps) * (tile + 6) + pad), BG)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), "同じ p、独立な乱数初期化（検証 seed）", fill=INK, font=font(18, True))
    for j, (name, p) in enumerate(zip(STAGES, ps)):
        y = 60 + j * (tile + 6)
        d.text((pad, y + tile // 2 - 20), name, fill=INK, font=font(15, True))
        d.text((pad, y + tile // 2 + 2), f"p = {p:.1f}", fill=MUTED, font=font(13))
        for i, s in enumerate(seeds):
            im = Image.open(root / f"seed{s}" / f"p{p:.2f}_seed{s}.png").convert("RGB").resize((tile, tile), Image.LANCZOS)
            sheet.paste(im, (lab_w + i * (tile + 6), y))
    for i, s in enumerate(seeds):
        d.text((lab_w + i * (tile + 6) + 2, 42), f"seed {s}", fill=MUTED, font=font(12))
    sheet.save(out)


def stage_strip(root: Path, out: Path, seed: int, ps, tile: int = 220, cols: int = 6) -> None:
    pad = 14
    rows = (len(ps) + cols - 1) // cols
    sheet = Image.new("RGB", (pad + cols * (tile + 6) + pad, 44 + rows * (tile + 24) + pad), BG)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), f"一様な p の独立な計算（探索 seed {seed}）: 段階の間の中間形態", fill=INK, font=font(18, True))
    for k, p in enumerate(ps):
        f = next(root.glob(f"seed{seed}_*/p{p:.2f}_seed{seed}.png"))
        x, y = pad + (k % cols) * (tile + 6), 44 + (k // cols) * (tile + 24)
        sheet.paste(Image.open(f).convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y + 18))
        d.text((x + 2, y), f"p = {p:.1f}", fill=MUTED, font=font(13))
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/pit_mouth")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/images")
    a = ap.parse_args()
    sweep_compare(a.artifacts / "sweep/sweep.png", a.out / "pit_mouth_sweep_compare.png")
    seeds_sheet(a.artifacts / "validation", a.out / "pit_mouth_seeds.png", (100, 101, 102, 103), (0.2, 1.0, 2.0, 3.1, 3.8))
    stage_strip(a.artifacts / "explore", a.out / "pit_mouth_stages.png", 1,
                (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 4.0))
    print("saved pit_mouth_sweep_compare.png, pit_mouth_seeds.png, pit_mouth_stages.png in", a.out)


if __name__ == "__main__":
    main()
