#!/usr/bin/env python3
"""Synthesize goal / anti-goal reference figures for pit_pattern_goal.md.

These are *hand-drawn programmatic targets*, not simulation output.
Feature width is held constant on purpose so the document can show
"length / branching change, width does not".

Example:
    uv run python src/pitpattern/figures/make_goal_figures.py
    make goal-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Canonical stroke width in pixels for all "good" goal figures.
LINE_WIDTH = 5
BG = 230  # light gray
FG = 20  # near black


def blank(h: int, w: int, value: int = BG) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def draw_disk(img: np.ndarray, cx: float, cy: float, radius: float) -> None:
    cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(radius)), FG, -1, cv2.LINE_AA)


def draw_segment(
    img: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    thickness: int = LINE_WIDTH,
) -> None:
    cv2.line(
        img,
        (int(round(x0)), int(round(y0))),
        (int(round(x1)), int(round(y1))),
        FG,
        thickness,
        cv2.LINE_AA,
    )


def draw_poly(
    img: np.ndarray,
    points: list[tuple[float, float]],
    *,
    thickness: int = LINE_WIDTH,
    closed: bool = False,
) -> None:
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    cv2.polylines(img, [pts], closed, FG, thickness, cv2.LINE_AA)


def panel_type_i(size: int = 220, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Regular round pits — Type I."""
    rng = rng or np.random.default_rng(0)
    img = blank(size, size)
    r = LINE_WIDTH / 2.0
    spacing = 28.0
    margin = 22.0
    y = margin
    row = 0
    while y < size - margin:
        x_off = (spacing * 0.5) if row % 2 else 0.0
        x = margin + x_off
        while x < size - margin:
            jitter = rng.normal(0.0, 0.8, size=2)
            draw_disk(img, x + jitter[0], y + jitter[1], r)
            x += spacing
        y += spacing * 0.866
        row += 1
    return img


def panel_type_iiil(size: int = 220, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Short tubular / slightly larger openings — Type IIIL. Same stroke width."""
    rng = rng or np.random.default_rng(1)
    img = blank(size, size)
    half = LINE_WIDTH / 2.0
    # Mix of short segments and a few round openings, all same width.
    segments = [
        (36, 40, 58, 48),
        (88, 38, 118, 42),
        (150, 44, 178, 52),
        (48, 78, 72, 92),
        (110, 80, 140, 78),
        (168, 86, 190, 100),
        (40, 120, 70, 128),
        (100, 118, 128, 134),
        (156, 124, 184, 120),
        (52, 160, 78, 172),
        (112, 158, 146, 166),
        (166, 162, 192, 178),
        (70, 196, 98, 190),
        (130, 194, 160, 200),
    ]
    for x0, y0, x1, y1 in segments:
        j = rng.normal(0.0, 1.2, size=4)
        draw_segment(img, x0 + j[0], y0 + j[1], x1 + j[2], y1 + j[3])
    # A few remaining round pits (same diameter as I).
    for cx, cy in ((34, 96), (198, 70), (92, 54), (186, 148), (30, 178)):
        draw_disk(img, cx, cy, half)
    return img


def panel_type_iv_b(size: int = 220, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Short branching trees — Type IV_B. Finite arms, constant width."""
    rng = rng or np.random.default_rng(2)
    img = blank(size, size)

    def tree(cx: float, cy: float, scale: float = 1.0, rot: float = 0.0) -> None:
        # Local Y / short forks around (cx, cy).
        arms = [
            (0.0, 0.0, 18.0, -6.0),
            (0.0, 0.0, -14.0, 12.0),
            (0.0, 0.0, 8.0, 16.0),
            (18.0, -6.0, 28.0, -18.0),
            (18.0, -6.0, 30.0, 2.0),
            (-14.0, 12.0, -26.0, 22.0),
            (8.0, 16.0, 4.0, 28.0),
        ]
        c, s = np.cos(rot), np.sin(rot)
        for x0, y0, x1, y1 in arms:
            p0x = cx + scale * (c * x0 - s * y0)
            p0y = cy + scale * (s * x0 + c * y0)
            p1x = cx + scale * (c * x1 - s * y1)
            p1y = cy + scale * (s * x1 + c * y1)
            draw_segment(img, p0x, p0y, p1x, p1y)

    centers = [
        (48, 48, 1.0, 0.2),
        (118, 42, 0.95, -0.4),
        (178, 56, 1.05, 0.6),
        (56, 112, 1.0, 1.1),
        (126, 108, 1.1, -0.9),
        (182, 120, 0.9, 0.3),
        (64, 172, 1.0, -0.2),
        (132, 168, 1.05, 0.8),
        (180, 178, 0.95, -0.7),
    ]
    for cx, cy, sc, rot in centers:
        jx, jy = rng.normal(0.0, 1.5, size=2)
        tree(cx + jx, cy + jy, sc, rot)
    return img


def draw_labyrinth(
    img: np.ndarray,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    row_pitch: float = 16.0,
    link_pitch: float = 34.0,
    amp: float = 5.0,
    thickness: int = LINE_WIDTH,
    rng: np.random.Generator | None = None,
) -> None:
    """Dense winding labyrinth / gyrus mesh with constant stroke width.

    Matches pptx slide 6 center and slide 7 IVV panel: connected maze,
    background remains as small islands — width does not blow up.
    """
    rng = rng or np.random.default_rng(3)
    rows: list[list[tuple[float, float]]] = []
    y = y0
    row_i = 0
    while y <= y1:
        pts: list[tuple[float, float]] = []
        x = x0
        yy = float(y)
        phase = row_i * 0.85
        while x <= x1:
            yy = float(y) + amp * np.sin(0.085 * x + phase) + float(rng.normal(0, 0.35))
            pts.append((x, yy))
            x += 10.0
        draw_poly(img, pts, thickness=thickness)
        rows.append(pts)
        y += row_pitch
        row_i += 1

    # Cross-links → single connected gyrus / labyrinth (not parallel stripes).
    for r in range(len(rows) - 1):
        a, b = rows[r], rows[r + 1]
        n = min(len(a), len(b))
        start = 2 + (r % 2) * 2
        for i in range(start, n - 2, max(int(link_pitch / 10), 2)):
            if rng.random() < 0.22:
                continue
            draw_segment(
                img,
                a[i][0],
                a[i][1],
                b[i][0] + float(rng.normal(0, 1.5)),
                b[i][1],
                thickness=thickness,
            )


def panel_type_iv_v(size: int = 220, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Dense labyrinth / gyrus network — Type IV_V. Same width as I–IV_B."""
    rng = rng or np.random.default_rng(3)
    img = blank(size, size)
    draw_labyrinth(
        img,
        x0=12,
        x1=size - 12,
        y0=18,
        y1=size - 16,
        row_pitch=15.5,
        link_pitch=28.0,
        amp=4.5,
        rng=rng,
    )
    return img


def make_schematic_panels(panel_size: int = 220, gap: int = 12) -> np.ndarray:
    panels = [
        panel_type_i(panel_size),
        panel_type_iiil(panel_size),
        panel_type_iv_b(panel_size),
        panel_type_iv_v(panel_size),
    ]
    labels = ["I", "IIIL", "IV_B", "IV_V"]
    label_h = 28
    h = panel_size + label_h
    w = panel_size * 4 + gap * 3
    canvas = blank(h, w, 255)
    x = 0
    for panel, label in zip(panels, labels):
        canvas[0:panel_size, x : x + panel_size] = panel
        cv2.putText(
            canvas,
            label,
            (x + panel_size // 2 - 18, panel_size + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            40,
            2,
            cv2.LINE_AA,
        )
        x += panel_size + gap
    return canvas


def make_target_sweep(height: int = 160, width: int = 720) -> np.ndarray:
    """One horizontal strip: I → IIIL → IV_B → IV_V with fixed stroke width."""
    img = blank(height, width)
    rng = np.random.default_rng(11)
    half = LINE_WIDTH / 2.0

    # --- Type I region (left) ---
    for row in range(4):
        y = 28 + row * 32
        x_off = 14 if row % 2 else 0
        x = 16 + x_off
        while x < width * 0.32:
            fade = 1.0 - (x / (width * 0.32))
            if rng.random() < 0.35 + 0.65 * fade:
                draw_disk(img, x + rng.normal(0, 0.5), y + rng.normal(0, 0.5), half)
            x += 26

    # --- IIIL short tubes (mid-left) ---
    for i, (y, length, ang) in enumerate(
        [
            (30, 22, 0.15),
            (52, 26, -0.2),
            (74, 20, 0.4),
            (96, 28, -0.1),
            (118, 22, 0.25),
            (140, 24, -0.35),
        ]
    ):
        for k in range(5):
            cx = width * 0.22 + k * 36 + (i % 2) * 10
            if cx > width * 0.52:
                continue
            dx = length * np.cos(ang + 0.1 * k)
            dy = length * np.sin(ang + 0.1 * k)
            draw_segment(img, cx - dx / 2, y - dy / 2, cx + dx / 2, y + dy / 2)

    # --- IV_B short branches (center) ---
    for i, cy in enumerate([34, 62, 90, 118, 142]):
        for j in range(4):
            cx = width * 0.42 + j * 42 + (i % 2) * 12
            if cx > width * 0.72:
                continue
            tree_arms = [
                (-10, 0, 12, -4),
                (-10, 0, -8, 12),
                (12, -4, 22, -14),
                (12, -4, 24, 4),
                (-8, 12, -18, 18),
            ]
            for x0, y0, x1, y1 in tree_arms:
                draw_segment(img, cx + x0, cy + y0, cx + x1, cy + y1)

    # --- IV_V dense labyrinth / gyrus (right) — NOT parallel leaf stripes ---
    draw_labyrinth(
        img,
        x0=width * 0.62,
        x1=width - 10,
        y0=18,
        y1=height - 14,
        row_pitch=15.0,
        link_pitch=26.0,
        amp=4.0,
        rng=rng,
    )

    # Phase labels under the strip (on a white footer).
    footer_h = 36
    out = blank(height + footer_h, width, 255)
    out[:height, :] = img
    for label, x in [("I", 40), ("IIIL", int(width * 0.28)), ("IV_B", int(width * 0.52)), ("IV_V", int(width * 0.82))]:
        cv2.putText(
            out,
            label,
            (x, height + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            40,
            2,
            cv2.LINE_AA,
        )
    # Constant-width callout.
    cv2.putText(
        out,
        f"stroke={LINE_WIDTH}px (constant)",
        (width - 220, height + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        80,
        1,
        cv2.LINE_AA,
    )
    return out


def anti_goal_parallel_stripes(size: int = 280) -> np.ndarray:
    """Parallel stripes without branching mesh — false IV_V (common model trap)."""
    img = blank(size, size)
    for i, y0 in enumerate(range(20, size - 12, 22)):
        pts = [(10, y0)]
        x = 10.0
        y = float(y0)
        while x < size - 10:
            x += 16
            y = float(y0) + 3.0 * np.sin(0.06 * x + i)
            pts.append((x, y))
        draw_poly(img, pts)
    return img


def anti_goal_dashes(size: int = 280) -> np.ndarray:
    """Scattered short dashes with no branching — false 'IV' by fragmentation."""
    img = blank(size, size)
    rng = np.random.default_rng(5)
    for _ in range(90):
        cx = float(rng.uniform(16, size - 16))
        cy = float(rng.uniform(16, size - 16))
        ang = float(rng.uniform(0, np.pi))
        length = float(rng.uniform(8, 18))
        dx = length * np.cos(ang)
        dy = length * np.sin(ang)
        draw_segment(img, cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2)
    return img


def anti_goal_thick(size: int = 280) -> np.ndarray:
    """Width blow-up — the failure mode this project is trying to avoid.

    Includes coarsened thick 'labyrinth-like' blobs: connectivity without
    constant feature width is still a fail.
    """
    img = blank(size, size)
    # Fat blobs and thick merged strokes.
    blobs = [
        (70, 70, 38),
        (160, 60, 48),
        (220, 120, 42),
        (80, 180, 50),
        (180, 200, 55),
    ]
    for cx, cy, r in blobs:
        cv2.circle(img, (cx, cy), r, FG, -1, cv2.LINE_AA)
    # Thick merged corridor (coarsened maze look).
    cv2.ellipse(img, (140, 140), (110, 28), 25, 0, 360, FG, -1, cv2.LINE_AA)
    cv2.ellipse(img, (150, 90), (90, 22), -15, 0, 360, FG, -1, cv2.LINE_AA)
    return img


def label_panel(img: np.ndarray, title: str) -> np.ndarray:
    h, w = img.shape[:2]
    out = blank(h + 32, w, 255)
    out[:h, :] = img
    cv2.putText(out, title, (8, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 40, 1, cv2.LINE_AA)
    return out


def write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"failed to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/images"),
        help="Directory for goal / anti-goal PNGs",
    )
    args = parser.parse_args()
    out = args.output_dir

    # Remove obsolete anti-goal name if present.
    obsolete = out / "anti_goal_labyrinth.png"
    if obsolete.exists():
        obsolete.unlink()

    write_png(out / "goal_schematic_i_iiil_ivb_ivv.png", make_schematic_panels())
    write_png(out / "goal_target_sweep.png", make_target_sweep())
    write_png(
        out / "anti_goal_parallel_stripes.png",
        label_panel(anti_goal_parallel_stripes(), "ANTI: parallel stripes (false IV_V)"),
    )
    write_png(out / "anti_goal_dashes.png", label_panel(anti_goal_dashes(), "ANTI: dashes, no branching"))
    write_png(out / "anti_goal_thick.png", label_panel(anti_goal_thick(), "ANTI: width blow-up"))

    print(f"Wrote goal figures to {out.resolve()}")
    for name in (
        "goal_schematic_i_iiil_ivb_ivv.png",
        "goal_target_sweep.png",
        "anti_goal_parallel_stripes.png",
        "anti_goal_dashes.png",
        "anti_goal_thick.png",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
