#!/usr/bin/env python3
"""Gray-Scott (f, k) path time-sweep -> pit-pattern phase-transition video.

The reaction-diffusion equations are the repository's unmodified Gray-Scott
model (same kernel, dA/dB and explicit Euler dt=1 as ``evaluate_pit_steps``):

    A_t = dA Lap(A) - A B^2 + f (1 - A)
    B_t = dB Lap(B) + A B^2 - (k + f) B      (+ weak additive noise on B)

What changes over time is only the position ``s`` on one straight parameter
path ``(f, k)(s) = P0 + s (P1 - P0)``.  ``s(t)`` is a piecewise-linear schedule
(knots ``(t_frac, s)``) so the sweep can dwell inside narrow windows such as
the finite-worm (Type III) regime.  The additive noise amplitude may also
follow a schedule ``sigma(t)``; the v2 preset uses a short strong burst while
ramping into the labyrinth window to erase the lattice orientation memory
(see ``--preset``).  Each frame is classified into
{uniform, spots, worm, labyrinth, holes} from the Euler number of the Otsu
binary image, component elongation and coverage, and mapped to Kudo-like
labels (Type I / Type III / Type IV branching / Type IV villous).

Example:
    python3 src/sweep_video.py --out artifacts/pit-sweep.mp4 \
        --phases-out artifacts/pit-sweep-phases.png --size 256 --frames 1500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pit_steps import KERNEL, make_initial_state, smooth_unit_noise  # noqa: E402


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# v1 schedule (t_frac, s), non-monotone.
#
# The hexagonal spot lattice formed at s=0 is metastable up to s~0.45, where
# the worm regime is already long-wormed, so v1 overshoots to s=0.47 (spots
# start to elongate) and then steps back to s=0.35, where the freshly formed
# stripes break up into short finite dashes ("quench and anneal").  Holes need
# s~0.94 to nucleate quickly; s>=1.0 floods to uniform B, so the sweep ends at 0.94.
SCHEDULE_V1: list[tuple[float, float]] = [
    (0.00, 0.00),
    (0.14, 0.00),  # Type I dwell (hex spots)
    (0.22, 0.47),
    (0.24, 0.47),  # overshoot: lattice destabilises into stripes
    (0.28, 0.35),
    (0.50, 0.35),  # Type III dwell (short worms / dashes)
    (0.62, 0.66),
    (0.74, 0.66),  # Type IV branching dwell (labyrinth)
    (0.84, 0.94),
    (1.00, 0.94),  # Type IV villous dwell (inverted hex holes)
]
NOISE_SCHEDULE_V1 = None  # constant sigma = 0.003

# v2 schedule: monotone s(t).  A direct ramp to s=0.40 followed by a dwell lets
# the lattice convert into short worms without the overshoot (6/6 seeds in
# internal/gs-sweep-experiments/exp7_*), especially when the Type I lattice is
# kept slightly disordered by stronger noise during its formation.
SCHEDULE_V2: list[tuple[float, float]] = [
    (0.00, 0.00),
    (0.12, 0.00),  # Type I dwell (spots, sigma=0.010 keeps the lattice disordered)
    (0.24, 0.40),
    (0.54, 0.40),  # Type III dwell (short worms; conversion takes ~10k steps)
    (0.64, 0.66),
    (0.76, 0.66),  # Type IV branching dwell (labyrinth)
    (0.85, 0.94),
    (1.00, 0.94),  # Type IV villous dwell (inverted hex holes)
]
# v2 noise sigma(t): strong noise only while ramping into the labyrinth window,
# which erases the lattice orientation memory so the labyrinth re-nucleates
# isotropically (meandering, gyrus-like) instead of as parallel stripes.
NOISE_SCHEDULE_V2: list[tuple[float, float]] = [
    (0.00, 0.010),
    (0.12, 0.010),
    (0.14, 0.003),
    (0.54, 0.003),
    (0.57, 0.020),
    (0.62, 0.020),
    (0.64, 0.003),
    (1.00, 0.003),
]
# Frames simulated under sigma above this are captioned "(transition)": the
# classifier is not meaningful on a noise-dominated field.
TRANSITION_SIGMA = 0.012
DEFAULT_SCHEDULE = SCHEDULE_V2
PRESETS = {
    "v1": (SCHEDULE_V1, NOISE_SCHEDULE_V1),
    "v2": (SCHEDULE_V2, NOISE_SCHEDULE_V2),
}


@dataclass
class SweepConfig:
    size: int = 256
    seed: int = 7
    seed_density: float = 0.035
    d_a: float = 1.0
    d_b: float = 0.5
    path_start: tuple[float, float] = (0.026, 0.061)
    path_end: tuple[float, float] = (0.040, 0.058)
    schedule: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_SCHEDULE)
    )
    total_steps: int = 120_000
    noise: float = 0.003
    # Optional time-varying noise sigma(t) as knots (t_frac, sigma); overrides ``noise``.
    noise_schedule: list[tuple[float, float]] | None = field(
        default_factory=lambda: list(NOISE_SCHEDULE_V2)
    )
    noise_seed: int = 11
    # Initial condition: Gaussian smoothing scale of the seed noise (None -> repo
    # default max(1.5, size/80)).  Smaller values give a more disordered seed field.
    init_smooth_sigma: float | None = None
    frames: int = 1500
    fps: int = 30
    scale: int = 3
    b_display_max: float = 0.38
    label_hold_frames: int = 4
    # Initial transient (seed blobs settling into spots) is shown as "forming".
    label_start_step: int = 2000

    def to_json(self) -> dict:
        d = asdict(self)
        d["path_start"] = list(self.path_start)
        d["path_end"] = list(self.path_end)
        d["schedule"] = [list(x) for x in self.schedule]
        if self.noise_schedule is not None:
            d["noise_schedule"] = [list(x) for x in self.noise_schedule]
        return d

    def sigma_at(self, step: int) -> float:
        if self.noise_schedule is None:
            return self.noise
        return schedule_s(step, self.total_steps, self.noise_schedule)


def parse_schedule(text: str) -> list[tuple[float, float]]:
    """``"0:0,0.2:0,0.4:0.5,1:1"`` -> [(0,0), (0.2,0), (0.4,0.5), (1,1)]."""
    knots = []
    for item in text.split(","):
        t, s = item.split(":")
        knots.append((float(t), float(s)))
    knots.sort()
    if knots[0][0] > 0.0:
        knots.insert(0, (0.0, knots[0][1]))
    if knots[-1][0] < 1.0:
        knots.append((1.0, knots[-1][1]))
    return knots


def schedule_s(step: int, total_steps: int, knots: list[tuple[float, float]]) -> float:
    t = step / max(total_steps, 1)
    ts = [k[0] for k in knots]
    ss = [k[1] for k in knots]
    return float(np.interp(t, ts, ss))


def path_fk(s: float, p0: tuple[float, float], p1: tuple[float, float]) -> tuple[float, float]:
    f = p0[0] + s * (p1[0] - p0[0])
    k = p0[1] + s * (p1[1] - p0[1])
    return f, k


# ---------------------------------------------------------------------------
# Gray-Scott step (unchanged equations)
# ---------------------------------------------------------------------------

def gs_step(
    a: np.ndarray,
    b: np.ndarray,
    f: float,
    k: float,
    d_a: float = 1.0,
    d_b: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    ab2 = a * b * b
    lap_a = cv2.filter2D(a, -1, KERNEL)
    lap_b = cv2.filter2D(b, -1, KERNEL)
    next_a = a + d_a * lap_a - ab2 + f * (1.0 - a)
    next_b = b + d_b * lap_b + ab2 - (k + f) * b
    return np.clip(next_a, 0.0, 1.0), np.clip(next_b, 0.0, 1.0)


def initial_state(cfg: "SweepConfig") -> tuple[np.ndarray, np.ndarray]:
    """Repo initial condition (``make_initial_state``) with optional custom smoothing scale."""
    if cfg.init_smooth_sigma is None:
        return make_initial_state(cfg.size, cfg.seed, cfg.seed_density)
    rng = np.random.default_rng(cfg.seed)
    h = w = cfg.size
    a = np.ones((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)
    noise = smooth_unit_noise(rng, (h, w), sigma=cfg.init_smooth_sigma)
    threshold = float(np.quantile(noise, 1.0 - cfg.seed_density))
    seeds = noise > threshold
    b[seeds] = 0.85
    a[seeds] = 0.25
    a += 0.02 * rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    b += 0.02 * rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    return np.clip(a, 0.0, 1.0), np.clip(b, 0.0, 1.0)


def add_noise(b: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0.0:
        return b
    return np.clip(b + sigma * rng.standard_normal(b.shape, dtype=np.float32), 0.0, 1.0)


# ---------------------------------------------------------------------------
# figure/ground and phase classification
# ---------------------------------------------------------------------------

PHASE_LABELS = {
    "forming": "(pattern forming)",
    "transition": "(transition: noise burst)",
    "uniform": "uniform (no pattern)",
    "spots": "Type I  (round pits)",
    "worm": "Type III  (tubular pits)",
    "labyrinth": "Type IV  branching",
    "holes": "Type IV  villous (inverted)",
}
PHASE_ORDER = ["spots", "worm", "labyrinth", "holes"]


def _components(mask: np.ndarray, connectivity: int, min_area: int):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    return int(keep.sum()), labels, stats[1:][keep]


def phase_metrics(b: np.ndarray) -> dict[str, float]:
    """Figure/ground statistics of the Otsu-binarised B field."""
    size = b.shape[0]
    min_area = max(4, (size // 64) ** 2)
    b_std = float(b.std())
    out: dict[str, float] = {"b_mean": float(b.mean()), "b_std": b_std}
    if b_std < 0.02:
        out.update(
            coverage=float(b.mean() > 0.1),
            n_fg=0, n_bg=0, euler=0.0, euler_ratio=0.0,
            largest_fg_fraction=0.0, aspect_median=1.0, phase_id="uniform",
        )
        return out

    b_u8 = np.clip(b * 255.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(b_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = b_u8 >= thr
    bg = ~fg
    n_fg, _, fg_stats = _components(fg, 8, min_area)
    n_bg, _, bg_stats = _components(bg, 4, min_area)
    euler = float(n_fg - n_bg)
    euler_ratio = euler / max(n_fg + n_bg, 1)
    coverage = float(fg.mean())

    if n_fg > 0:
        fg_areas = fg_stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        largest_fg_fraction = float(fg_areas.max() / fg_areas.sum())
    else:
        largest_fg_fraction = 0.0

    out.update(
        coverage=coverage, n_fg=n_fg, n_bg=n_bg, euler=euler,
        euler_ratio=euler_ratio, largest_fg_fraction=largest_fg_fraction,
        aspect_median=_aspect_median(fg, min_area) if n_fg > 0 else 1.0,
        dark_aspect_median=_aspect_median(bg, min_area) if n_bg > 0 else 1.0,
    )
    out["phase_id"] = classify_phase(out)
    return out


def _aspect_median(mask: np.ndarray, min_area: int) -> float:
    """Area-weighted median of minAreaRect aspect ratios of the mask's components."""
    aspects: list[float] = []
    weights: list[float] = []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        lo, hi = min(w, h), max(w, h)
        aspects.append(hi / max(lo, 1.0))
        weights.append(area)
    if not aspects:
        return 1.0
    order = np.argsort(aspects)
    cw = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(aspects)[order][np.searchsorted(cw, cw[-1] / 2)])


def classify_phase(m: dict[str, float]) -> str:
    if m.get("b_std", 1.0) < 0.02:
        return "uniform"
    r = m["euler_ratio"]
    if r < -0.5 and m["n_bg"] >= 6 and m.get("dark_aspect_median", 1.0) < 1.6:
        # Round dark islands on a bright connected ground: figure/ground inverted.
        # (A bright network with elongated dark grooves is still a labyrinth.)
        return "holes"
    if r > 0.5 and m["largest_fg_fraction"] < 0.35:
        return "worm" if m["aspect_median"] >= 1.9 else "spots"
    return "labyrinth"


class LabelSmoother:
    """A new label must persist ``hold`` frames before the caption switches."""

    def __init__(self, hold: int = 4):
        self.hold = hold
        self.current: str | None = None
        self.candidate: str | None = None
        self.count = 0

    def update(self, label: str) -> str:
        if self.current is None:
            self.current = label
            return label
        if label == self.current:
            self.candidate = None
            self.count = 0
            return self.current
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate = label
            self.count = 1
        if self.count >= self.hold:
            self.current = label
            self.candidate = None
            self.count = 0
        return self.current


# ---------------------------------------------------------------------------
# rendering (crystal-violet look: bright pits on dark ground)
# ---------------------------------------------------------------------------

_CV_ANCHORS = [  # (position, (R, G, B))
    (0.00, (22, 6, 40)),
    (0.30, (70, 30, 110)),
    (0.60, (150, 95, 190)),
    (0.85, (222, 200, 240)),
    (1.00, (250, 246, 255)),
]


def crystal_violet_lut() -> np.ndarray:
    xs = np.linspace(0.0, 1.0, 256)
    pos = [p for p, _ in _CV_ANCHORS]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for ch in range(3):
        vals = [c[ch] for _, c in _CV_ANCHORS]
        lut[:, ch] = np.clip(np.interp(xs, pos, vals), 0, 255).astype(np.uint8)
    return lut[:, ::-1]  # BGR for OpenCV


_LUT = crystal_violet_lut()


def render_b(b: np.ndarray, b_max: float, scale: int) -> np.ndarray:
    v = np.clip(b / max(b_max, 1e-6), 0.0, 1.0) ** 0.8
    idx = (v * 255).astype(np.uint8)
    img = _LUT[idx]
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def compose_frame(
    img: np.ndarray,
    *,
    step: int,
    s: float,
    f: float,
    k: float,
    label: str,
    caption_height: int = 96,
) -> np.ndarray:
    h, w = img.shape[:2]
    frame = np.zeros((h + caption_height, w, 3), dtype=np.uint8)
    frame[:h] = img
    bar = frame[h:]
    bar[:] = (28, 20, 34)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, PHASE_LABELS.get(label, label), (14, 34), font, 0.85, (255, 240, 250), 2, cv2.LINE_AA)
    cv2.putText(bar, f"s = {s:.3f}    f = {f:.4f}    k = {k:.4f}    t = {step}",
                (14, 70), font, 0.6, (200, 190, 215), 1, cv2.LINE_AA)
    # progress bar along the path
    x0, x1, y = w - 250, w - 44, 22
    cv2.rectangle(bar, (x0, y - 5), (x1, y + 5), (90, 70, 110), 1)
    xs = int(x0 + s * (x1 - x0))
    cv2.rectangle(bar, (x0, y - 4), (xs, y + 4), (210, 150, 255), -1)
    cv2.putText(bar, "P0", (x0 - 28, y + 5), font, 0.45, (170, 160, 190), 1, cv2.LINE_AA)
    cv2.putText(bar, "P1", (x1 + 6, y + 5), font, 0.45, (170, 160, 190), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# simulation driver
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    frame: int
    step: int
    s: float
    f: float
    k: float
    phase_raw: str
    phase: str
    metrics: dict[str, float]


def run_sweep(
    cfg: SweepConfig,
    *,
    on_frame=None,
    log_every: int = 100,
) -> list[FrameRecord]:
    """Run the time sweep; call ``on_frame(record, b)`` for every frame."""
    a, b = initial_state(cfg)
    rng = np.random.default_rng(cfg.noise_seed)
    frame_steps = np.linspace(0, cfg.total_steps, cfg.frames + 1, dtype=int)[1:]
    frame_set = {int(x): i for i, x in enumerate(frame_steps)}
    smoother = LabelSmoother(cfg.label_hold_frames)
    records: list[FrameRecord] = []
    t0 = time.time()
    for step in range(1, cfg.total_steps + 1):
        s = schedule_s(step, cfg.total_steps, cfg.schedule)
        f, k = path_fk(s, cfg.path_start, cfg.path_end)
        a, b = gs_step(a, b, f, k, cfg.d_a, cfg.d_b)
        b = add_noise(b, cfg.sigma_at(step), rng)
        if step in frame_set:
            m = phase_metrics(b)
            raw = str(m.pop("phase_id"))
            if step < cfg.label_start_step:
                label = "forming"
            elif cfg.sigma_at(step) > TRANSITION_SIGMA:
                label = "transition"
                smoother.current = None  # re-classify freshly once the burst ends
            else:
                label = smoother.update(raw)
            rec = FrameRecord(frame_set[step], step, s, f, k, raw, label, m)
            records.append(rec)
            if on_frame is not None:
                on_frame(rec, b)
            if log_every and rec.frame % log_every == 0:
                print(
                    f"frame {rec.frame:5d} step {step:7d} s={s:.3f} f={f:.4f} k={k:.4f} "
                    f"{label:9s} nfg={m['n_fg']:3.0f} nbg={m['n_bg']:3.0f} "
                    f"cov={m['coverage']:.2f} asp={m['aspect_median']:.2f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    return records


def representative_frames(records: list[FrameRecord]) -> dict[str, int]:
    """Middle frame of the longest run of each phase label."""
    best: dict[str, tuple[int, int]] = {}
    i = 0
    while i < len(records):
        j = i
        while j + 1 < len(records) and records[j + 1].phase == records[i].phase:
            j += 1
        label = records[i].phase
        length = j - i + 1
        if label not in best or length > best[label][0]:
            best[label] = (length, (i + j) // 2)
        i = j + 1
    return {label: idx for label, (_, idx) in best.items()}


def write_phase_sheet(
    frames: dict[str, np.ndarray],
    records: list[FrameRecord],
    rep: dict[str, int],
    path: Path,
    tile: int = 384,
) -> None:
    tiles = []
    for label in PHASE_ORDER:
        if label not in rep:
            continue
        rec = records[rep[label]]
        img = cv2.resize(frames[label], (tile, tile), interpolation=cv2.INTER_AREA)
        header = np.zeros((64, tile, 3), dtype=np.uint8)
        header[:] = (28, 20, 34)
        cv2.putText(header, PHASE_LABELS[label], (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 240, 250), 1, cv2.LINE_AA)
        cv2.putText(header, f"s={rec.s:.2f} f={rec.f:.4f} k={rec.k:.4f}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 190, 215), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([header, img], axis=0))
    if not tiles:
        return
    sep = np.full((tiles[0].shape[0], 6, 3), (28, 20, 34), dtype=np.uint8)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [sep, t]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(row, axis=1))


def write_metrics_csv(records: list[FrameRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["frame", "step", "s", "f", "k", "phase_raw", "phase"] + sorted(records[0].metrics)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in records:
            row = {"frame": r.frame, "step": r.step, "s": r.s, "f": r.f, "k": r.k,
                   "phase_raw": r.phase_raw, "phase": r.phase}
            row.update(r.metrics)
            w.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_pair(text: str) -> tuple[float, float]:
    f, k = text.split(",")
    return float(f), float(k)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = SweepConfig()
    p.add_argument("--out", type=Path, required=True, help="output .mp4 path")
    p.add_argument("--phases-out", type=Path, help="4-phase representative frame sheet (.png)")
    p.add_argument("--metrics-out", type=Path, help="per-frame metrics (.csv)")
    p.add_argument("--config", type=Path, help="JSON file with SweepConfig fields (overrides defaults)")
    p.add_argument("--size", type=int, default=d.size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--steps", type=int, default=d.total_steps, help="total simulation steps")
    p.add_argument("--frames", type=int, default=d.frames)
    p.add_argument("--fps", type=int, default=d.fps)
    p.add_argument("--scale", type=int, default=d.scale, help="integer upscale of the field for display")
    p.add_argument("--noise", type=float, default=d.noise, help="additive noise sigma on B per step")
    p.add_argument("--path-start", type=parse_pair, default=d.path_start, metavar="f,k")
    p.add_argument("--path-end", type=parse_pair, default=d.path_end, metavar="f,k")
    p.add_argument("--preset", choices=sorted(PRESETS), default="v2",
                   help="named (schedule, noise schedule) pair; v1 = non-monotone overshoot, v2 = monotone + noise burst")
    p.add_argument("--schedule", type=str, help='knots "t:s,t:s,..." (t,s in [0,1]); flat segments = dwell')
    p.add_argument("--noise-schedule", type=str,
                   help='knots "t:sigma,..." for time-varying noise; "const" = use --noise everywhere')
    p.add_argument("--init-smooth-sigma", type=float, help="smoothing scale of the seed noise (default: repo value)")
    p.add_argument("--b-max", type=float, default=d.b_display_max, help="B value mapped to the brightest colour")
    p.add_argument("--hold", type=int, default=d.label_hold_frames, help="frames a new label must persist")
    p.add_argument("--crf", type=int, default=22, help="libx264 constant rate factor (lower = larger/better)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> SweepConfig:
    cfg = SweepConfig(
        size=args.size, seed=args.seed, total_steps=args.steps, frames=args.frames,
        fps=args.fps, scale=args.scale, noise=args.noise, path_start=args.path_start,
        path_end=args.path_end, b_display_max=args.b_max, label_hold_frames=args.hold,
        init_smooth_sigma=args.init_smooth_sigma,
    )
    cfg.schedule, cfg.noise_schedule = PRESETS[args.preset]
    cfg.schedule = list(cfg.schedule)
    cfg.noise_schedule = None if cfg.noise_schedule is None else list(cfg.noise_schedule)
    if args.schedule:
        cfg.schedule = parse_schedule(args.schedule)
    if args.noise_schedule:
        cfg.noise_schedule = None if args.noise_schedule == "const" else parse_schedule(args.noise_schedule)
    if args.config:
        data = json.loads(Path(args.config).read_text())
        for key, value in data.items():
            if key in ("path_start", "path_end"):
                value = tuple(value)
            elif key in ("schedule", "noise_schedule") and value is not None:
                value = [tuple(x) for x in value]
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    print(json.dumps(cfg.to_json(), indent=1))

    import imageio.v2 as imageio

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.out), fps=cfg.fps, codec="libx264", quality=None,
        pixelformat="yuv420p", macro_block_size=16,
        ffmpeg_params=["-crf", str(args.crf), "-preset", "medium"],
    )
    rep_frames: dict[str, np.ndarray] = {}
    raw_by_frame: dict[int, np.ndarray] = {}

    def on_frame(rec: FrameRecord, b: np.ndarray) -> None:
        img = render_b(b, cfg.b_display_max, cfg.scale)
        frame = compose_frame(img, step=rec.step, s=rec.s, f=rec.f, k=rec.k, label=rec.phase)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raw_by_frame[rec.frame] = img

    records = run_sweep(cfg, on_frame=on_frame)
    writer.close()

    rep = representative_frames(records)
    for label, idx in rep.items():
        rep_frames[label] = raw_by_frame[records[idx].frame]
    present = [PHASE_LABELS[p] for p in PHASE_ORDER if p in rep]
    print("phases present:", present)
    for label in PHASE_ORDER:
        rs = [r.s for r in records if r.phase == label]
        if rs:
            print(f"  {label:9s} s in [{min(rs):.3f}, {max(rs):.3f}]  frames={len(rs)}")
    if args.phases_out:
        write_phase_sheet(rep_frames, records, rep, args.phases_out)
    if args.metrics_out:
        write_metrics_csv(records, args.metrics_out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
