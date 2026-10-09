#!/usr/bin/env python3
"""Evaluate pit-pattern progression sweeps against docs/pit_pattern_goal.md.

Runs parameter sweeps, classifies I / IIIL / IV_B / IV_V, checks temporal
stability, and writes goal reports plus contact sheets.

Example:
    uv run python src/pitpattern/evaluation/evaluate_pit_goal.py sweep --output artifacts/pit_goal_sweep
    uv run python src/pitpattern/evaluation/evaluate_pit_goal.py ramp --output artifacts/pit_goal_ramp
    uv run python src/pitpattern/evaluation/evaluate_pit_goal.py robustness --output artifacts/pit_goal_robust
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pitpattern.metrics.evaluate_pit_steps import write_metrics
from pitpattern.metrics.pit_goal_metrics import (
    classify_morphology,
    evaluate_goal_sequence,
    measure_goal_pattern,
    measure_temporal_stability,
)
from pitpattern.models.simulate_pit_buckling import BucklingConfig, pit_field, run_buckling
from pitpattern.models.simulate_pit_progression import (
    ProgressionModel,
    ProgressionSettings,
    run_ramp_batch,
    run_sweep,
    simulate_at_p,
    write_contact_sheet,
)


def _write_goal_report(output: Path, report: dict[str, Any]) -> None:
    with (output / "goal_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def _write_timeseries_csv(output: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output / "timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_progression_sweep(
    *,
    model: ProgressionModel,
    settings: ProgressionSettings,
    output: Path,
    p_count: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    p_values = np.linspace(0.0, 1.0, p_count)
    rows = run_sweep(model=model, settings=settings, output=output, p_values=p_values)

    for row in rows:
        row["model"] = model.value

    ts_rows: list[dict[str, Any]] = []
    for p in p_values:
        tag = f"p_{p:.3f}"
        ts_path = output / tag / "timeseries.json"
        if ts_path.exists():
            with ts_path.open(encoding="utf-8") as handle:
                for entry in json.load(handle):
                    entry["p"] = float(p)
                    entry["model"] = model.value
                    ts_rows.append(entry)

    report = evaluate_goal_sequence(rows)
    report["model"] = model.value
    report["settings"] = asdict(settings)
    report["p_values"] = [float(p) for p in p_values]

    write_metrics(output, rows)
    _write_timeseries_csv(output, ts_rows)
    _write_goal_report(output, report)

    images = [output / f"p_{row['p']:.3f}" / "final_annotated.png" for row in rows]
    write_contact_sheet(rows, images, output / "contact_sheet.png")

    summary_lines = [
        f"model={model.value} passed={report['passed']}",
        f"phases={' -> '.join(report['seen_phases'])}",
        f"width_ok={report['width_ok']} monotonic={report['monotonic']}",
    ]
    with (output / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    return report


def run_progression_ramp(
    *,
    model: ProgressionModel,
    settings: ProgressionSettings,
    output: Path,
    ramp_rates: list[float],
    reverse: bool,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows = run_ramp_batch(
        model=model,
        settings=settings,
        output=output,
        ramp_rates=ramp_rates,
        reverse=reverse,
    )
    for row in rows:
        row["model"] = model.value
        row["reverse"] = reverse

    write_metrics(output, rows)
    report = {
        "model": model.value,
        "reverse": reverse,
        "ramp_rates": ramp_rates,
        "rows": rows,
        "labels": [row.get("morphology", classify_morphology(row)) for row in rows],
    }
    _write_goal_report(output, report)
    return report


def run_buckling_baseline_sweep(
    *,
    size: int,
    steps: int,
    seed: int,
    output: Path,
) -> list[dict[str, Any]]:
    """Map buckling growth as a pseudo-p axis for comparison (not the main goal)."""
    output.mkdir(parents=True, exist_ok=True)
    growth_values = np.linspace(0.08, 0.22, 12)
    rows: list[dict[str, Any]] = []
    for growth in growth_values:
        config = BucklingConfig(
            name=f"baseline_g_{growth:.3f}",
            description="Buckling baseline sweep over growth.",
            wavelength=12.0,
            quadratic=max(0.0, 1.0 - growth * 3.0),
            growth=float(growth),
        )
        result = run_buckling(
            config,
            size=size,
            steps=steps,
            dt=0.35,
            seed=seed,
            snapshot_count=4,
        )
        b = pit_field(result.u)
        metrics = measure_goal_pattern(b, field_kind="activator")
        label = classify_morphology(metrics)
        tag = f"g_{growth:.3f}"
        case_dir = output / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case_dir / "final.png"), cv2.applyColorMap(
            cv2.normalize(result.u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_BONE,
        ))
        row = {
            "p": float(growth),
            "growth": float(growth),
            "model": "buckling_baseline",
            "morphology": label,
            **metrics,
        }
        rows.append(row)
    write_metrics(output, rows)
    return rows


def run_robustness(
    *,
    model: ProgressionModel,
    output: Path,
    seeds: list[int],
    sizes: list[int],
    p_count: int = 11,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    all_reports: list[dict[str, Any]] = []

    for size in sizes:
        for seed in seeds:
            tag = f"size_{size}_seed_{seed}"
            settings = ProgressionSettings(size=size, steps=2500, seed=seed, snapshot_count=5)
            case_dir = output / tag
            report = run_progression_sweep(
                model=model,
                settings=settings,
                output=case_dir,
                p_count=p_count,
            )
            report["seed"] = seed
            report["size"] = size
            all_reports.append(report)

    passed = sum(1 for r in all_reports if r.get("passed"))
    aggregate = {
        "model": model.value,
        "seeds": seeds,
        "sizes": sizes,
        "cases": len(all_reports),
        "passed_cases": passed,
        "pass_rate": passed / max(len(all_reports), 1),
        "reports": all_reports,
    }
    _write_goal_report(output, aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["sweep", "ramp", "reverse_ramp", "compare", "robustness", "smoke"],
    )
    parser.add_argument(
        "--model",
        choices=[m.value for m in ProgressionModel],
        default=ProgressionModel.SH_INHIBITOR.value,
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_goal"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wavelength", type=float, default=12.0)
    parser.add_argument("--p-count", type=int, default=25)
    parser.add_argument(
        "--ramp-rates",
        type=float,
        nargs="+",
        default=[1.0 / 6000.0, 1.0 / 4000.0, 1.0 / 2500.0],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 17, 23])
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 128])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ProgressionSettings(
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        wavelength=args.wavelength,
    )
    model = ProgressionModel(args.model)

    if args.command == "smoke":
        settings = ProgressionSettings(size=96, steps=800, seed=args.seed, snapshot_count=3)
        for p in (0.0, 0.4, 0.7, 1.0):
            result = simulate_at_p(p, model=model, settings=settings)
            metrics = measure_goal_pattern(result.u, field_kind="height")
            print(
                f"p={p:.1f} morphology={classify_morphology(metrics)} "
                f"branch={metrics['branch_density']:.3f} "
                f"width={metrics['feature_width_median']:.1f}"
            )
        return

    if args.command == "sweep":
        report = run_progression_sweep(
            model=model,
            settings=settings,
            output=args.output,
            p_count=args.p_count,
        )
        print(json.dumps({k: report[k] for k in ("passed", "seen_phases", "width_ok")}, indent=2))
        return

    if args.command == "ramp":
        run_progression_ramp(
            model=model,
            settings=settings,
            output=args.output,
            ramp_rates=args.ramp_rates,
            reverse=False,
        )
        return

    if args.command == "reverse_ramp":
        run_progression_ramp(
            model=model,
            settings=settings,
            output=args.output,
            ramp_rates=args.ramp_rates,
            reverse=True,
        )
        return

    if args.command == "compare":
        args.output.mkdir(parents=True, exist_ok=True)
        reports: dict[str, Any] = {}
        for prog_model in ProgressionModel:
            model_dir = args.output / prog_model.value
            reports[prog_model.value] = run_progression_sweep(
                model=prog_model,
                settings=settings,
                output=model_dir,
                p_count=min(args.p_count, 15),
            )
        reports["buckling_baseline"] = run_buckling_baseline_sweep(
            size=args.size,
            steps=args.steps,
            seed=args.seed,
            output=args.output / "buckling_baseline",
        )
        _write_goal_report(args.output, reports)
        return

    if args.command == "robustness":
        aggregate = run_robustness(
            model=model,
            output=args.output,
            seeds=args.seeds,
            sizes=args.sizes,
            p_count=min(args.p_count, 11),
        )
        print(
            f"robustness: {aggregate['passed_cases']}/{aggregate['cases']} passed "
            f"({aggregate['pass_rate']:.0%})"
        )
        return


if __name__ == "__main__":
    main()
