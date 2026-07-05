#!/usr/bin/env python3
"""Interactive simulator for the local-inhibitor pit model (A, B, H).

Uses evaluate_pit_steps.calc_step with feed/k, optional reaction saturation,
and inhibitor field H only. No mask, environment C, or dynamic-env modes.

Example:
    uv run python src/exp2.py
    uv run python src/exp2.py --preset step4_pit_iv_inhibitor
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pygame
import pygame_gui
from pygame.locals import QUIT

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluate_pit_steps import (  # noqa: E402
    DEFAULT_EVAL_RUN,
    EvalRunSettings,
    ModelConfig,
    build_suite,
    calc_step,
    load_eval_case,
    make_initial_polarity,
    make_initial_state,
    measure_pattern,
    render_skeleton_overlay,
    uses_polarity,
)

# Type IV metrics shown live in the UI (key, short label, decimals).
LIVE_METRICS: tuple[tuple[str, str, int], ...] = (
    ("branch_density", "branch", 3),
    ("longest_component_fraction", "longest", 3),
    ("long_line_score", "long", 2),
    ("mean_segment_length", "mean_seg", 1),
    ("branch_per_component", "br/comp", 1),
    ("binary_coverage", "cov", 3),
    ("skeleton_components", "skel_cmp", 0),
)

# 3 columns × 2 rows, landscape.
COLS = 3
ROWS = 2
VIEW_W = 426
VIEW_H = 360
SCREEN_W = COLS * VIEW_W
SCREEN_H = ROWS * VIEW_H

SNAPSHOT_DIR = Path("artifacts/exp2_snapshots")
DEFAULT_PRESET = "step4_pit_iv_inhibitor"

EXP2_PRESETS: tuple[str, ...] = (
    "step0_pit_iv_baseline",
    "step4_pit_iv_inhibitor",
    "step5_pit_iv_bistable_width",
    "step5b_pit_iv_bistable_inhibitor",
    "step5c_pit_iv_three_channel",
)

PRESET_MENU_LABELS: dict[str, str] = {
    "step0_pit_iv_baseline": "step0 baseline",
    "step4_pit_iv_inhibitor": "step4 H",
    "step5_pit_iv_bistable_width": "step5 W",
    "step5b_pit_iv_bistable_inhibitor": "step5b W+H",
    "step5c_pit_iv_three_channel": "step5c W+H+M",
}

_CORE = ("feed", "k", "d_a", "d_b")
_H = (
    "inhibitor_strength",
    "inhibitor_diffusion",
    "inhibitor_source",
    "inhibitor_decay",
)
_W = ("bistable_strength", "bistable_threshold")
_M = ("coverage_feedback", "coverage_target")

# Sliders shown per preset (only parameters that matter for that variant).
PRESET_PARAM_ATTRS: dict[str, tuple[str, ...]] = {
    "step0_pit_iv_baseline": _CORE,
    "step4_pit_iv_inhibitor": _CORE + _H,
    "step5_pit_iv_bistable_width": _CORE + _W,
    "step5b_pit_iv_bistable_inhibitor": _CORE + _W + _H,
    "step5c_pit_iv_three_channel": _CORE + _W + _H + _M,
}

# (ModelConfig attr, label, min, max, default, decimals)
PARAM_SPECS: list[tuple[str, str, float, float, float, int]] = [
    ("feed", "feed", 0.02, 0.12, 0.033, 3),
    ("k", "k", 0.02, 0.12, 0.056, 3),
    ("d_a", "dA", 0.1, 2.0, 1.0, 2),
    ("d_b", "dB", 0.1, 2.0, 0.5, 2),
    ("inhibitor_strength", "H gamma", 0.0, 0.2, 0.05, 3),
    ("inhibitor_diffusion", "H D", 0.0, 1.0, 0.6, 2),
    ("inhibitor_source", "H rho", 0.0, 0.2, 0.05, 3),
    ("inhibitor_decay", "H delta", 0.0, 0.3, 0.10, 3),
    ("bistable_strength", "W lambda", 0.0, 0.4, 0.0, 3),
    ("bistable_threshold", "W beta", 0.05, 0.6, 0.22, 2),
    ("coverage_feedback", "M kappa", 0.0, 0.3, 0.0, 3),
    ("coverage_target", "M phi*", 0.0, 0.6, 0.30, 2),
]

PARAM_SPEC_BY_ATTR = {spec[0]: spec for spec in PARAM_SPECS}
SLIDER_ATTRS = {spec[0] for spec in PARAM_SPECS}

# Control strip geometry (bottom row).
PANEL_PAD = 10
PANEL_HEADER_H = 118
SLIDER_ROW_H = 34
SLIDER_COLS = 3


class Viewport:
    def __init__(self, x0: int, y0: int, x1: int, y1: int, grid_w: int, grid_h: int):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.gw = x1 - x0
        self.gh = y1 - y0
        self.sw = self.gw / grid_w
        self.sh = self.gh / grid_h

    def contains(self, gx: int, gy: int) -> bool:
        return self.x0 <= gx < self.x1 and self.y0 <= gy < self.y1

    def to_grid(self, gx: int, gy: int) -> tuple[int, int]:
        """Map screen (gx, gy) to OpenCV/numpy center (col, row)."""
        col = int((gx - self.x0) / self.sw)
        row = int((gy - self.y0) / self.sh)
        return col, row

    @property
    def origin(self) -> tuple[int, int]:
        return self.x0, self.y0

    def blit_rgb(self, screen: pygame.Surface, rgb: np.ndarray) -> None:
        surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        screen.blit(pygame.transform.scale(surf, (self.gw, self.gh)), self.origin)

    def caption(self, screen: pygame.Surface, text: str, font: pygame.font.Font) -> None:
        x, y = self.origin
        for line in text.splitlines():
            screen.blit(font.render(line, True, (240, 240, 240)), (x + 6, y + 6))
            y += 18


def field_to_u8(field: np.ndarray, hi_percentile: float = 99.0) -> np.ndarray:
    lo = float(field.min())
    hi = float(np.percentile(field, hi_percentile))
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((field - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def param_attrs_for_case(case_name: str, base_config: ModelConfig) -> tuple[str, ...]:
    if case_name in PRESET_PARAM_ATTRS:
        return PRESET_PARAM_ATTRS[case_name]
    # custom: show sliders for the preset we branched from
    if base_config.name in PRESET_PARAM_ATTRS:
        return PRESET_PARAM_ATTRS[base_config.name]
    attrs: list[str] = list(_CORE)
    if base_config.inhibitor_strength > 0 or base_config.inhibitor_source > 0:
        attrs.extend(_H)
    if base_config.bistable_strength > 0:
        attrs.extend(_W)
    if base_config.coverage_feedback > 0:
        attrs.extend(_M)
    return tuple(dict.fromkeys(attrs))


def exp2_suite() -> dict[str, ModelConfig]:
    all_steps = {cfg.name: cfg for cfg in build_suite("steps")}
    return {name: all_steps[name] for name in EXP2_PRESETS if name in all_steps}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-inhibitor pit pattern simulator (exp2)."
    )
    parser.add_argument(
        "--preset",
        choices=EXP2_PRESETS,
        default=DEFAULT_PRESET,
        help=f"Built-in case (default: {DEFAULT_PRESET}).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="metrics.json from evaluate_pit_steps (requires --case).",
    )
    parser.add_argument("--case", help="Case name inside metrics.json.")
    parser.add_argument("--size", type=int, default=DEFAULT_EVAL_RUN.size)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_RUN.seed)
    parser.add_argument("--seed-density", type=float, default=DEFAULT_EVAL_RUN.seed_density)
    parser.add_argument(
        "--metrics-frames",
        type=int,
        default=60,
        help="Recompute skeleton metrics every N display frames (default: 60).",
    )
    return parser.parse_args(argv)


def resolve_startup(args: argparse.Namespace) -> tuple[ModelConfig, EvalRunSettings]:
    settings = EvalRunSettings(
        size=args.size, seed=args.seed, seed_density=args.seed_density
    )
    if args.metrics is not None:
        if not args.case:
            raise SystemExit("--case is required with --metrics")
        config, metrics_settings = load_eval_case(args.metrics, args.case)
        if args.size == DEFAULT_EVAL_RUN.size:
            settings = EvalRunSettings(
                size=metrics_settings.size,
                seed=(
                    args.seed
                    if args.seed != DEFAULT_EVAL_RUN.seed
                    else metrics_settings.seed
                ),
                seed_density=(
                    args.seed_density
                    if args.seed_density != DEFAULT_EVAL_RUN.seed_density
                    else metrics_settings.seed_density
                ),
            )
        return config, settings

    suite = exp2_suite()
    if args.preset not in suite:
        raise SystemExit(f"unknown preset {args.preset!r}")
    return suite[args.preset], settings


class InhibitorSimulator:
    def __init__(
        self,
        config: ModelConfig,
        settings: EvalRunSettings,
        *,
        metrics_frame_interval: int = 60,
    ) -> None:
        self.settings = settings
        self.w = self.h = settings.size

        self.view_ab = Viewport(0, 0, VIEW_W, VIEW_H, self.w, self.h)
        self.view_skeleton = Viewport(VIEW_W, 0, 2 * VIEW_W, VIEW_H, self.w, self.h)
        self.view_h = Viewport(2 * VIEW_W, 0, SCREEN_W, VIEW_H, self.w, self.h)
        self.screen_size = (SCREEN_W, SCREEN_H)

        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Pit pattern — exp2 (3×2)")
        self.font = pygame.font.Font(None, 26)
        self.font_small = pygame.font.Font(None, 22)
        self.clock = pygame.time.Clock()
        self.ui = pygame_gui.UIManager(self.screen_size)

        self.suite = exp2_suite()
        self.active_case = config.name
        self.base_config = config
        self.visible_param_attrs: tuple[str, ...] = ()
        self.paused = False
        self.step_count = 0
        self.steps_per_frame = 10
        self.metrics_frame_interval = max(1, metrics_frame_interval)
        self.frame_count = 0
        self._metrics_at_frame = -1
        self.live_metrics: dict[str, float] = {}
        self.skeleton_rgb: np.ndarray | None = None

        self.grid_a = np.zeros((self.h, self.w), dtype=np.float32)
        self.grid_b = np.zeros((self.h, self.w), dtype=np.float32)
        self.grid_c = np.zeros((self.h, self.w), dtype=np.float32)
        self.grid_h = np.zeros((self.h, self.w), dtype=np.float32)
        self.grid_r = np.zeros((self.h, self.w), dtype=np.float32)
        self.grid_p_x = np.zeros((self.h, self.w), dtype=np.float32)
        self.grid_p_y = np.zeros((self.h, self.w), dtype=np.float32)

        self.param_labels: dict[str, pygame_gui.elements.UILabel] = {}
        self.param_sliders: dict[str, pygame_gui.elements.UIHorizontalSlider] = {}
        self.param_decimals: dict[str, int] = {}

        self._build_panel()
        self.apply_config(config, reset=True)

    def _panel_rect(self) -> pygame.Rect:
        return pygame.Rect(
            PANEL_PAD,
            VIEW_H + PANEL_PAD,
            SCREEN_W - 2 * PANEL_PAD,
            VIEW_H - 2 * PANEL_PAD,
        )

    def _build_panel(self) -> None:
        panel = self._panel_rect()
        ox, oy = panel.x, panel.y
        pw = panel.width

        menu_options = ["custom"] + [
            PRESET_MENU_LABELS.get(name, name) for name in self.suite.keys()
        ]
        start_label = PRESET_MENU_LABELS.get(self.active_case, self.active_case)
        self.preset_menu = pygame_gui.elements.UIDropDownMenu(
            options_list=menu_options,
            starting_option=start_label,
            relative_rect=pygame.Rect(ox, oy, min(220, pw // 3), 28),
            manager=self.ui,
        )
        self._preset_label_to_name = {
            PRESET_MENU_LABELS.get(name, name): name for name in self.suite
        }

        bx = ox + min(228, pw // 3) + 8
        bw = 72
        self.btn_pause = pygame_gui.elements.UIButton(
            pygame.Rect(bx, oy, bw, 28), "Pause", self.ui
        )
        self.btn_step = pygame_gui.elements.UIButton(
            pygame.Rect(bx + bw + 6, oy, bw, 28), "Step", self.ui
        )
        self.btn_reset = pygame_gui.elements.UIButton(
            pygame.Rect(bx + 2 * (bw + 6), oy, bw, 28), "Reset", self.ui
        )
        self.btn_snapshot = pygame_gui.elements.UIButton(
            pygame.Rect(bx + 3 * (bw + 6), oy, bw + 16, 28), "Snapshot", self.ui
        )

        self.param_area_y = oy + PANEL_HEADER_H
        self.param_area_w = pw
        self.param_area_x = ox

        for attr, label, lo, hi, default, decimals in PARAM_SPECS:
            self.param_decimals[attr] = decimals
            self.param_labels[attr] = pygame_gui.elements.UILabel(
                pygame.Rect(0, 0, 100, 14),
                f"{label}: {default:.{decimals}f}",
                manager=self.ui,
            )
            self.param_sliders[attr] = pygame_gui.elements.UIHorizontalSlider(
                pygame.Rect(0, 0, 100, 14),
                default,
                (lo, hi),
                manager=self.ui,
            )
            self.param_labels[attr].hide()
            self.param_sliders[attr].hide()

    def _layout_param_sliders(self) -> None:
        attrs = list(self.visible_param_attrs)
        n = len(attrs)
        if n == 0:
            return

        cols = min(SLIDER_COLS, n)
        col_w = (self.param_area_w - (cols - 1) * 12) // cols
        slider_w = col_w - 8
        rows = (n + cols - 1) // cols

        for index, attr in enumerate(attrs):
            col = index % cols
            row = index // cols
            x = self.param_area_x + col * (col_w + 12)
            y = self.param_area_y + row * SLIDER_ROW_H
            label = self.param_labels[attr]
            slider = self.param_sliders[attr]
            label.set_relative_position((x, y))
            label.set_dimensions((slider_w, 14))
            slider.set_relative_position((x, y + 16))
            slider.set_dimensions((slider_w, 14))

        used_h = rows * SLIDER_ROW_H
        max_h = VIEW_H - PANEL_HEADER_H - 2 * PANEL_PAD
        if used_h > max_h:
            # Fallback: 4 columns if still tight (step5c = 12 sliders → 3 rows)
            cols = min(4, n)
            col_w = (self.param_area_w - (cols - 1) * 10) // cols
            slider_w = col_w - 8
            for index, attr in enumerate(attrs):
                col = index % cols
                row = index // cols
                x = self.param_area_x + col * (col_w + 10)
                y = self.param_area_y + row * SLIDER_ROW_H
                self.param_labels[attr].set_relative_position((x, y))
                self.param_labels[attr].set_dimensions((slider_w, 14))
                self.param_sliders[attr].set_relative_position((x, y + 16))
                self.param_sliders[attr].set_dimensions((slider_w, 14))

    def _sync_param_visibility(self) -> None:
        self.visible_param_attrs = param_attrs_for_case(
            self.active_case, self.base_config
        )
        visible = set(self.visible_param_attrs)
        for attr in SLIDER_ATTRS:
            label = self.param_labels[attr]
            slider = self.param_sliders[attr]
            if attr in visible:
                label.show()
                slider.show()
            else:
                label.hide()
                slider.hide()
        self._layout_param_sliders()

    def _slider(self, attr: str) -> float:
        return float(self.param_sliders[attr].get_current_value())

    def _set_slider(self, attr: str, value: float) -> None:
        slider = self.param_sliders[attr]
        lo, hi = slider.value_range
        slider.set_current_value(min(hi, max(lo, value)))
        dec = self.param_decimals[attr]
        label_text = PARAM_SPEC_BY_ATTR[attr][1]
        self.param_labels[attr].set_text(f"{label_text}: {self._slider(attr):.{dec}f}")

    def model_config(self) -> ModelConfig:
        kwargs: dict[str, float] = {}
        for field in fields(ModelConfig):
            if field.name not in SLIDER_ATTRS:
                continue
            if field.name in self.visible_param_attrs:
                kwargs[field.name] = self._slider(field.name)
            else:
                kwargs[field.name] = getattr(self.base_config, field.name)
        return ModelConfig(
            name=self.active_case,
            description="exp2 live",
            **kwargs,
        )

    def apply_config(self, config: ModelConfig, *, reset: bool = False) -> None:
        self.base_config = config
        self.active_case = config.name
        if config.name in self.suite:
            menu_label = PRESET_MENU_LABELS.get(config.name, config.name)
            self.preset_menu.selected_option = menu_label
        self._sync_param_visibility()
        for field in fields(ModelConfig):
            if field.name in SLIDER_ATTRS:
                self._set_slider(field.name, getattr(config, field.name))
        if reset:
            self.reset_fields()

    def reset_all(self) -> None:
        """Reset fields and parameters to the active base preset."""
        self.apply_config(self.base_config, reset=True)

    def refresh_metrics(self, *, force: bool = False) -> None:
        if (
            not force
            and self._metrics_at_frame >= 0
            and self.frame_count - self._metrics_at_frame < self.metrics_frame_interval
        ):
            return
        self.live_metrics = measure_pattern(self.grid_a, self.grid_b, self.grid_c)
        self.skeleton_rgb = render_skeleton_overlay(self.grid_b)[:, :, ::-1]
        self._metrics_at_frame = self.frame_count

    def metrics_lines(self) -> list[str]:
        if not self.live_metrics:
            return ["metrics: computing..."]
        lines: list[str] = []
        for key, label, decimals in LIVE_METRICS:
            value = self.live_metrics.get(key, 0.0)
            if decimals == 0:
                lines.append(f"{label} {value:.0f}")
            else:
                lines.append(f"{label} {value:.{decimals}f}")
        row_a = "  ".join(lines[0:3])
        row_b = "  ".join(lines[3:6])
        row_c = "  ".join(lines[6:])
        return [row_a, row_b, row_c]

    def reset_fields(self) -> None:
        cfg = self.model_config()
        self.grid_a, self.grid_b = make_initial_state(
            self.h, self.settings.seed, self.settings.seed_density
        )
        self.grid_c.fill(0.0)
        self.grid_h.fill(0.0)
        self.grid_r.fill(0.0)
        if uses_polarity(cfg):
            self.grid_p_x, self.grid_p_y = make_initial_polarity(
                self.h, self.settings.seed
            )
        else:
            self.grid_p_x.fill(0.0)
            self.grid_p_y.fill(0.0)
        self.step_count = 0
        self.frame_count = 0
        self.refresh_metrics(force=True)

    def simulation_step(self) -> None:
        self.grid_a, self.grid_b, self.grid_c, self.grid_h, self.grid_r, self.grid_p_x, self.grid_p_y = calc_step(
            self.grid_a,
            self.grid_b,
            self.grid_c,
            self.grid_h,
            self.grid_r,
            self.grid_p_x,
            self.grid_p_y,
            self.model_config(),
        )
        self.step_count += 1

    def save_snapshot(self) -> Path:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = SNAPSHOT_DIR / f"snapshot_{stamp}.png"
        pygame.image.save(self.screen, path)
        return path

    def _config_caption_lines(self, cfg: ModelConfig) -> list[str]:
        lines = [
            f"feed {cfg.feed:.4f}  k {cfg.k:.4f}",
            f"dA {cfg.d_a:.2f}  dB {cfg.d_b:.2f}",
        ]
        if cfg.inhibitor_strength > 0 or cfg.inhibitor_source > 0:
            lines.append(
                f"H γ {cfg.inhibitor_strength:.3f}  D {cfg.inhibitor_diffusion:.2f}  "
                f"ρ {cfg.inhibitor_source:.3f}  δ {cfg.inhibitor_decay:.3f}"
            )
        if cfg.bistable_strength > 0:
            lines.append(
                f"W λ {cfg.bistable_strength:.3f}  β {cfg.bistable_threshold:.2f}"
            )
        if cfg.coverage_feedback > 0:
            lines.append(
                f"M κ {cfg.coverage_feedback:.3f}  φ* {cfg.coverage_target:.2f}"
            )
        return lines

    def _paint_ab_seed(self) -> None:
        if self.ui.get_hovering_any_element():
            return
        if not any(pygame.mouse.get_pressed()):
            return
        mx, my = pygame.mouse.get_pos()
        if not self.view_ab.contains(mx, my):
            return
        pos = self.view_ab.to_grid(mx, my)
        if pygame.mouse.get_pressed()[0]:
            cv2.circle(self.grid_a, pos, 8, 0.0, -1)
            cv2.circle(self.grid_b, pos, 8, 1.0, -1)
        if pygame.mouse.get_pressed()[2]:
            cv2.circle(self.grid_a, pos, 8, 1.0, -1)
            cv2.circle(self.grid_b, pos, 8, 0.0, -1)

    def _render(self) -> None:
        self.screen.fill((0, 0, 0))

        ab = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        ab[:, :, 0] = field_to_u8(self.grid_a)
        ab[:, :, 2] = field_to_u8(self.grid_b)
        self.view_ab.blit_rgb(self.screen, ab)
        short_case = PRESET_MENU_LABELS.get(self.active_case, self.active_case)
        self.view_ab.caption(
            self.screen,
            (
                f"A (R)  B (B)  |  {short_case}\n"
                f"FPS {self.clock.get_fps():.1f}  steps {self.step_count}  "
                f"seed {self.settings.seed}"
            ),
            self.font_small,
        )

        if self.skeleton_rgb is not None:
            self.view_skeleton.blit_rgb(self.screen, self.skeleton_rgb)
        skel_header = "skeleton  white=skel  red=3-arm fork  cyan=end"
        self.view_skeleton.caption(
            self.screen,
            "\n".join([skel_header, *self.metrics_lines()]),
            self.font_small,
        )

        h_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        h_img[:, :, 0] = field_to_u8(self.grid_h)
        self.view_h.blit_rgb(self.screen, h_img)
        cfg = self.model_config()
        self.view_h.caption(
            self.screen,
            "inhibitor H\n" + "\n".join(self._config_caption_lines(cfg)),
            self.font_small,
        )

        # Bottom control strip label
        panel = self._panel_rect()
        hint = self.font_small.render(
            "controls (preset shows only related sliders)",
            True,
            (120, 120, 120),
        )
        self.screen.blit(hint, (panel.x, panel.y + 32))

    def _resolve_preset_name(self, menu_text: str) -> str | None:
        if menu_text == "custom":
            return None
        if menu_text in self.suite:
            return menu_text
        return self._preset_label_to_name.get(menu_text)

    def _on_ui(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_pause:
                self.paused = not self.paused
                self.btn_pause.set_text("Run" if self.paused else "Pause")
                if self.paused:
                    self.refresh_metrics(force=True)
            elif event.ui_element == self.btn_step:
                self.simulation_step()
                self.refresh_metrics(force=True)
            elif event.ui_element == self.btn_reset:
                self.reset_all()
            elif event.ui_element == self.btn_snapshot:
                print(f"saved {self.save_snapshot()}", flush=True)

        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element != self.preset_menu:
                return
            if event.text == "custom":
                self.active_case = "custom"
                self._sync_param_visibility()
                return
            preset_name = self._resolve_preset_name(event.text)
            if preset_name and preset_name in self.suite:
                self.apply_config(self.suite[preset_name], reset=True)

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element not in self.param_sliders.values():
                return
            self.active_case = "custom"
            self.preset_menu.selected_option = "custom"
            for attr, slider in self.param_sliders.items():
                if event.ui_element == slider and attr in self.visible_param_attrs:
                    self._set_slider(attr, self._slider(attr))
                    break

    def _on_key(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_ESCAPE, pygame.K_e):
            self.reset_all()
        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused
            self.btn_pause.set_text("Run" if self.paused else "Pause")
            if self.paused:
                self.refresh_metrics(force=True)
        elif event.key == pygame.K_s:
            self.simulation_step()
            self.refresh_metrics(force=True)

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            self.frame_count += 1
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                self.ui.process_events(event)
                self._on_ui(event)
                self._on_key(event)

            self._paint_ab_seed()
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.simulation_step()
            self.refresh_metrics()
            self._render()
            self.ui.update(dt)
            self.ui.draw_ui(self.screen)
            pygame.display.update()
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config, settings = resolve_startup(args)
    InhibitorSimulator(
        config, settings, metrics_frame_interval=args.metrics_frames
    ).run()


if __name__ == "__main__":
    main()
