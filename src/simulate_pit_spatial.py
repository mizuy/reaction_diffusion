#!/usr/bin/env python3
"""Spatial-gradient pit progression: one run, p(x) left→right.

Unlike independent p-sweeps (which jump between attractors), this module runs
a *single* simulation where coefficients vary smoothly across the domain.
Left = Type I, right = Type IV_V, with intermediate morphologies in between.

Example:
    uv run python src/simulate_pit_spatial.py --output artifacts/pit_spatial
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pit_fission import graded_fission_mask
from pit_goal_metrics import (
    classify_spatial_morphology,
    measure_goal_pattern,
    measure_horizontal_runs,
    field_to_binary,
)
from simulate_pit_progression import ProgressionSettings, render_height


P_FISSION_OFF = 0.78


def fission_length_limit(p: float, *, wavelength: float = 12.0) -> float:
    """Max skeleton length (px) a pit unit may reach before crypt fission.

    This is the single knob that turns length into a function of progression:
    I keeps round pits, IIIL allows short tubes (~2-3x width), IV_B allows
    small branched trees, and above ``P_FISSION_OFF`` fission stops entirely
    so the IV_V labyrinth mesh can connect.
    """
    if p >= P_FISSION_OFF:
        return float("inf")
    base = 0.66 * wavelength  # ≈ spot skeleton + margin: keeps Type I round
    return float(base * np.exp(3.0 * max(p - 0.25, 0.0)))


@dataclass(frozen=True)
class SpatialFields:
    """Position-dependent coefficients derived from p(x, y)."""

    p_map: np.ndarray
    growth: np.ndarray
    quadratic: np.ndarray
    inhibitor_source: np.ndarray
    inhibitor_strength: np.ndarray
    inhibitor_decay: float = 0.12


def make_p_map(size: int, *, axis: str = "x", margin: float = 0.10) -> np.ndarray:
    """Smooth progression coordinate in [0, 1] along horizontal axis."""
    if axis != "x":
        raise ValueError("only axis='x' supported for now")
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    # cosine feather: flat near 0 and 1, smooth ramp in between
    t = np.clip((x - margin) / max(1.0 - 2 * margin, 1.0e-6), 0.0, 1.0)
    p = 0.5 * (1.0 - np.cos(np.pi * t))
    return np.broadcast_to(p[None, :], (size, size)).astype(np.float32)


def fields_from_p_map(p_map: np.ndarray) -> SpatialFields:
    """Map p(x) to local coefficients — avoid global phase jump.

    IV_V is the isotropic labyrinth / gyrus phase of Swift-Hohenberg at low
    quadratic term.  No anisotropy / stripe-alignment forcing: the previous
    villous_elong / stripe_bias terms produced parallel stripes, which the
    goal document rejects as false IV_V.
    """
    p = p_map.astype(np.float32)

    # Spots (hexagons) at low p; quadratic -> ~0 at high p so the isotropic
    # labyrinth phase takes over on the right side of the field.
    quadratic = 1.06 * (1.0 - np.clip((p - 0.18) / 0.70, 0.0, 1.0) ** 1.2)

    growth = 0.265 - 0.03 * p

    # No anisotropic elongation forcing anywhere: IIIL emerges naturally as
    # the mixed spot/short-worm state where the quadratic term weakens, and
    # any directional term imprints parallel stripes downstream (false IV_V).

    # Inhibitor: only in the IV_B window — keeps branching units finite there
    # (short trees instead of long worms), then releases so the mesh may
    # connect into the IV_V labyrinth.
    inh_band = np.exp(-((p - 0.52) / 0.13) ** 2)
    inhibitor_strength = 0.52 * inh_band
    inhibitor_source = 0.26 * inh_band

    return SpatialFields(
        p_map=p,
        growth=growth,
        quadratic=quadratic,
        inhibitor_source=inhibitor_source,
        inhibitor_strength=inhibitor_strength,
    )


def _laplacian(u: np.ndarray) -> np.ndarray:
    return (
        np.roll(u, 1, 0)
        + np.roll(u, -1, 0)
        + np.roll(u, 1, 1)
        + np.roll(u, -1, 1)
        - 4.0 * u
    ).astype(np.float32)


def _sh_operator(u: np.ndarray, k0: float) -> np.ndarray:
    lap = _laplacian(u)
    s = u + lap / (k0**2)
    lap_s = _laplacian(s)
    return (s + lap_s / (k0**2)).astype(np.float32)


@dataclass
class SpatialResult:
    u: np.ndarray
    aux: np.ndarray
    fields: SpatialFields
    snapshots: list[tuple[int, np.ndarray]]


def simulate_spatial(
    settings: ProgressionSettings,
    *,
    rng: np.random.Generator | None = None,
    fission_interval: int = 20,
    fission_warmup: int = 600,
    fission_strength: float = 1.0,
) -> SpatialResult:
    size = settings.size
    if rng is None:
        rng = np.random.default_rng(settings.seed)

    u = (0.04 * rng.normal(0.0, 1.0, (size, size))).astype(np.float32)
    aux = np.zeros((size, size), dtype=np.float32)
    fields = fields_from_p_map(make_p_map(size))
    k0 = 2.0 * np.pi / settings.wavelength
    dt = settings.dt

    freqs = np.fft.fftfreq(size, d=1.0) * (2.0 * np.pi)
    kx = np.broadcast_to(freqs[None, :], (size, size)).astype(np.float32)
    ky = np.broadcast_to(freqs[:, None], (size, size)).astype(np.float32)
    k2 = kx * kx + ky * ky
    linear_base = 0.25 - (1.0 - k2 / (k0**2)) ** 2
    denom = np.clip(1.0 - dt * linear_base, 1.0e-4, None).astype(np.float32)

    snapshot_steps = set(
        np.linspace(0, settings.steps, settings.snapshot_count, dtype=int).tolist()
    )
    snapshots: list[tuple[int, np.ndarray]] = []
    if 0 in snapshot_steps:
        snapshots.append((0, u.copy()))

    for step in range(1, settings.steps + 1):
        pos = np.clip(u, 0.0, None)

        aux = (
            aux
            + dt * (fields.inhibitor_source * pos**2 - fields.inhibitor_decay * aux)
        ).astype(np.float32)
        aux = np.clip(aux + 0.06 * dt * _laplacian(aux), 0.0, 5.0)

        nonlinear = fields.quadratic * (u**2) - (u**3)
        nonlinear = nonlinear - fields.inhibitor_strength * aux * pos

        growth_corr = (fields.growth - 0.25) * u
        combined = nonlinear + growth_corr

        u_hat = np.fft.fft2(u)
        n_hat = np.fft.fft2(combined)
        u = np.real(np.fft.ifft2((u_hat + dt * n_hat) / denom)).astype(np.float32)
        u = np.clip(u, -2.5, 2.5).astype(np.float32)

        # Crypt fission: neck-and-split any unit longer than L(p).  This is
        # the explicit length-selection mechanism SH itself lacks: the mixed
        # spot/stripe state has no finite-length attractor, so without it the
        # IIIL band grows long worms.
        if fission_interval > 0 and step >= fission_warmup and step % fission_interval == 0:
            binary = field_to_binary(u, field_kind="height")
            mask = graded_fission_mask(
                binary,
                fields.p_map,
                max_length_at_p=lambda p: fission_length_limit(
                    p, wavelength=settings.wavelength
                ),
            )
            if bool(mask.any()):
                u = (u - fission_strength * mask).astype(np.float32)

        if step in snapshot_steps:
            snapshots.append((step, u.copy()))

    return SpatialResult(u=u, aux=aux, fields=fields, snapshots=snapshots)


def measure_spatial_strips(
    u: np.ndarray,
    n_strips: int = 8,
) -> list[dict]:
    """Metrics in vertical strips left→right.

    Each strip is measured on a window twice the strip width (centered on the
    strip, clipped at the field edges): single-strip windows are too narrow
    relative to the pattern wavelength for stable topology metrics.
    """
    h, w = u.shape
    strip_w = w // n_strips
    win_w = min(2 * strip_w, w)
    rows: list[dict] = []
    for i in range(n_strips):
        x0 = i * strip_w
        x1 = w if i == n_strips - 1 else (i + 1) * strip_w
        cx = (x0 + x1) // 2
        # Fixed-width window (shifted inward at the edges) so densities are
        # comparable across strips.
        win0 = int(np.clip(cx - win_w // 2, 0, w - win_w))
        patch = u[:, win0 : win0 + win_w]
        metrics = measure_goal_pattern(patch, field_kind="height")
        binary = field_to_binary(patch, field_kind="height")
        metrics.update(measure_horizontal_runs(binary))
        metrics["strip_width"] = float(patch.shape[1])
        metrics["component_density"] = float(
            metrics.get("component_count", 0.0) / max(patch.size, 1) * 1000.0
        )
        p_mean = float(i + 0.5) / n_strips
        rows.append(
            {
                "strip": i,
                "p_mean": p_mean,
                "x0": x0,
                "x1": x1,
                "morphology": classify_spatial_morphology(metrics),
                **metrics,
            }
        )
    return rows


def _normalize_u8_global(field: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((field - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def evaluate_spatial_continuity(u: np.ndarray, n_strips: int = 8) -> dict:
    """Check adjacent strips for smooth metric/image change."""
    h, w = u.shape
    strip_w = w // n_strips
    strips = measure_spatial_strips(u, n_strips=n_strips)

    lo = float(np.percentile(u, 1))
    hi = float(np.percentile(u, 99))
    overlap = max(4, strip_w // 5)

    img_diffs: list[float] = []
    metric_jumps: list[float] = []
    labels = [s["morphology"] for s in strips]

    for i in range(n_strips - 1):
        x_bound = (i + 1) * strip_w
        pa = u[:, max(0, x_bound - overlap):x_bound]
        pb = u[:, x_bound:min(w, x_bound + overlap)]
        sw = min(pa.shape[1], pb.shape[1])
        if sw <= 0:
            continue
        pa_u8 = _normalize_u8_global(pa[:, -sw:], lo, hi)
        pb_u8 = _normalize_u8_global(pb[:, :sw], lo, hi)
        img_diffs.append(float(np.mean(np.abs(pa_u8.astype(float) - pb_u8.astype(float)))))

        ma = strips[i]
        mb = strips[i + 1]
        dj = abs(ma.get("p90_component_elongation", 0) - mb.get("p90_component_elongation", 0))
        dj += abs(ma.get("branch_arm_density", 0) - mb.get("branch_arm_density", 0)) * 50
        metric_jumps.append(dj)

    phase_order = ["I", "IIIL", "IV_B", "IV_V"]
    order_idx = {n: j for j, n in enumerate(phase_order)}

    def _seen_and_monotonic(labs: list[str]) -> tuple[list[str], bool]:
        seen_: list[str] = []
        for lab in labs:
            if lab in ("fail", "empty"):
                continue
            if not seen_ or lab != seen_[-1]:
                seen_.append(lab)
        mono = True
        last = -1
        for lab in seen_:
            idx = order_idx.get(lab, -1)
            if idx >= 0 and idx < last:
                mono = False
            if idx >= 0:
                last = max(last, idx)
        return seen_, mono

    seen, monotonic = _seen_and_monotonic(labels)

    coarse = measure_spatial_strips(u, n_strips=4)
    coarse_labels = [s["morphology"] for s in coarse]
    coarse_seen, coarse_monotonic = _seen_and_monotonic(coarse_labels)

    max_img_jump = max(img_diffs) if img_diffs else 0.0

    # Continuity per goal doc §5.2: every strip must be a goal phase, and
    # adjacent strips must be the same or neighbouring phases (no skipping,
    # e.g. IIIL next to IV_V without an IV_B strip in between).  Pixel-level
    # image diffs are reported for reference but do not gate: even a perfect
    # pattern has large pixel diffs between phase-shifted adjacent bands.
    smooth = all(lab not in ("fail", "empty") for lab in labels)
    if smooth:
        for lab_a, lab_b in zip(labels[:-1], labels[1:]):
            ia = order_idx.get(lab_a, -1)
            ib = order_idx.get(lab_b, -1)
            if abs(ib - ia) > 1:
                smooth = False
                break
    phases_present = {lab for lab in labels if lab not in ("fail", "empty")}
    required = {"I", "IV_B", "IV_V"}
    optional = {"IIIL"}

    return {
        "passed": bool(
            required.issubset(phases_present)
            and monotonic
            and smooth
        ),
        "seen_phases": seen,
        "phases_present": sorted(phases_present),
        "iiil_present": bool(phases_present & optional),
        "strip_labels": labels,
        "coarse_labels": coarse_labels,
        "monotonic": bool(coarse_monotonic),
        "max_adjacent_image_diff": max_img_jump,
        "mean_adjacent_image_diff": float(np.mean(img_diffs)) if img_diffs else 0.0,
        "smooth": bool(smooth),
        "strips": strips,
    }


def render_spatial_overview(result: SpatialResult, output: Path) -> None:
    """One-page image: field + p map + strip metrics."""
    u = result.u
    h, w = u.shape
    height_img = render_height(u)

    # p-map colour bar (horizontal gradient)
    p_vis = (result.fields.p_map * 255).astype(np.uint8)
    p_bar = cv2.applyColorMap(p_vis, cv2.COLORMAP_TURBO)
    p_bar = cv2.resize(p_bar, (w, 24), interpolation=cv2.INTER_AREA)

    report = evaluate_spatial_continuity(u, n_strips=8)
    strip_h = 48
    strip_row = np.zeros((strip_h, w, 3), dtype=np.uint8)
    n = len(report["strips"])
    sw = w // n
    for i, s in enumerate(report["strips"]):
        x0 = i * sw
        cv2.rectangle(strip_row, (x0, 0), (x0 + sw - 1, strip_h - 1), (40, 40, 40), 1)
        cv2.putText(
            strip_row,
            s["morphology"],
            (x0 + 4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            strip_row,
            f"p~{s['p_mean']:.2f}",
            (x0 + 4, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

    title_h = 56
    title = np.zeros((title_h, w, 3), dtype=np.uint8)
    cv2.putText(
        title,
        "Spatial p(x): ONE run, left=I -> right=IV_V",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    phases = " -> ".join(report["seen_phases"])
    cv2.putText(
        title,
        f"phases: {phases}  |  max_adj_diff={report['max_adjacent_image_diff']:.1f}  passed={report['passed']}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    overview = np.vstack([title, height_img, p_bar, strip_row])
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overview)
    cv2.imwrite(str(output.with_name("spatial_height.png")), height_img)


def save_spatial_result(result: SpatialResult, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = evaluate_spatial_continuity(result.u, n_strips=8)

    cv2.imwrite(str(output / "final_height.png"), render_height(result.u))
    render_spatial_overview(result, output / "spatial_overview.png")

    with (output / "spatial_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                k: v
                for k, v in report.items()
                if k != "strips"
            }
            | {"strips": report["strips"]},
            f,
            indent=2,
        )

    with (output / "fields.json").open("w", encoding="utf-8") as f:
        json.dump(
            {k: float(v) if np.ndim(v) == 0 else "2d" for k, v in asdict(result.fields).items()},
            f,
            indent=2,
        )

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/pit_spatial"))
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--wavelength", type=float, default=12.0)
    p.add_argument("--snapshot-count", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = ProgressionSettings(
        size=args.size,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        wavelength=args.wavelength,
        snapshot_count=args.snapshot_count,
    )
    result = simulate_spatial(settings)
    report = save_spatial_result(result, args.output)
    print(json.dumps({k: report[k] for k in ("passed", "seen_phases", "max_adjacent_image_diff", "smooth")}, indent=2))


if __name__ == "__main__":
    main()
