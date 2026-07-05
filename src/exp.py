# -*- coding: utf-8 -*-
"""Interactive pit-pattern simulator with pygame_gui controls."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
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
    advance_reaction_diffusion,
    build_static_environment,
    build_suite,
    calc_step,
    effective_parameters,
    load_eval_case,
    make_initial_polarity,
    make_initial_state,
    uses_polarity,
)

DISP_CELL = 480
DEFAULT_GRID_SIZE = DEFAULT_EVAL_RUN.size
SNAPSHOT_DIR = Path("artifacts/exp_snapshots")
DEFAULT_METRICS_PATH = Path("artifacts/pit_step_eval/metrics.json")

# (attr, label, min, max, default, decimals)
PARAM_SPECS: list[tuple[str, str, float, float, float, int]] = [
    ("d_a", "dA", 0.1, 2.0, 1.0, 2),
    ("d_b", "dB", 0.1, 2.0, 0.5, 2),
    ("feed", "feed", 0.02, 0.12, 0.037, 3),
    ("k", "k", 0.02, 0.12, 0.058, 3),
    ("reaction_saturation", "react sat", 0.0, 3.0, 0.0, 2),
    ("cubic_damping", "B^3 damp", 0.0, 0.05, 0.0, 4),
    ("a_depletion", "A depl", 0.0, 1.0, 0.0, 2),
    ("uniform_core_decay", "core decay", 0.0, 0.5, 0.0, 3),
    ("static_env_scale", "env scale", 0.0, 2.0, 0.0, 2),
    ("feed_env_sensitivity", "feed sens", 0.0, 0.01, 0.0, 4),
    ("k_env_sensitivity", "k sens", -0.01, 0.01, 0.0, 4),
    ("env_rate", "env rate", 0.0, 0.1, 0.0, 3),
    ("env_diffusion", "env D", 0.0, 1.0, 0.0, 2),
    ("env_source", "env src", 0.0, 2.0, 0.0, 2),
    ("env_decay", "env decay", 0.0, 0.5, 0.0, 2),
    ("inhibitor_strength", "inhib gamma", 0.0, 1.0, 0.0, 2),
    ("inhibitor_diffusion", "inhib D", 0.0, 1.0, 0.0, 2),
    ("inhibitor_source", "inhib src", 0.0, 0.5, 0.0, 3),
    ("inhibitor_decay", "inhib decay", 0.0, 0.3, 0.0, 3),
    ("spatial_feed_delta", "mask dF", 0.0, 0.02, 0.0, 4),
    ("spatial_k_delta", "mask dK", -0.02, 0.0, 0.0, 4),
]

LEGACY_PRESETS: list[tuple[str, float, float]] = [
    ("legacy regular pit", 0.037, 0.058),
    ("legacy coral", 0.055, 0.062),
    ("legacy irregular", 0.042, 0.055),
]


def blit_text(surface, text, pos, font, color=None):
    if color is None:
        color = pygame.Color("white")
    words = [word.split(" ") for word in text.splitlines()]
    space = font.size(" ")[0]
    max_width, _ = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 1, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]
                y += word_height
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]
        y += word_height


class Geometry:
    def __init__(self, x0, y0, x1, y1, w, h):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.gw = x1 - x0
        self.gh = y1 - y0
        self.lw = w
        self.lh = h
        self.sw = self.gw / self.lw
        self.sh = self.gh / self.lh

    def within_global(self, gx, gy):
        return self.x0 <= gx < self.x1 and self.y0 <= gy < self.y1

    def get_local(self, gx, gy):
        return int((gx - self.x0) / self.sw), int((gy - self.y0) / self.sh)

    @property
    def pos(self):
        return self.x0, self.y0

    @property
    def size(self):
        return self.gw, self.gh

    def blit(self, screen, image):
        surf = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
        surf = pygame.transform.scale(surf, self.size)
        screen.blit(surf, self.pos)

    def text(self, screen, text, pos, font, color=None):
        if color is None:
            color = pygame.Color("white")
        pos = (self.pos[0] + pos[0], self.pos[1] + pos[1])
        blit_text(screen, text, pos, font, color)


def parse_exp_args(argv: list[str] | None = None) -> argparse.Namespace:
    suite_names = [cfg.name for cfg in build_suite("all")]
    parser = argparse.ArgumentParser(
        description="Interactive pit-pattern experiment (reproduces evaluate_pit_steps runs)."
    )
    parser.add_argument(
        "--preset",
        choices=suite_names,
        help="Load a built-in evaluate_pit_steps case (model + eval IC).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="metrics.json from evaluate_pit_steps (use with --case).",
    )
    parser.add_argument(
        "--case",
        help="Case name inside metrics.json (e.g. step1_pit_iv_saturated).",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_GRID_SIZE, help="Grid width/height.")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_RUN.seed, help="RNG seed (eval default: 7).")
    parser.add_argument(
        "--seed-density",
        type=float,
        default=DEFAULT_EVAL_RUN.seed_density,
        help="Fraction of stochastic seed pixels (eval default: 0.035).",
    )
    parser.add_argument(
        "--no-eval-ic",
        action="store_true",
        help="Use manual center seed instead of evaluate_pit_steps make_initial_state.",
    )
    return parser.parse_args(argv)


class DiffusionReaction:
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        eval_settings: EvalRunSettings | None = None,
        eval_repro: bool = True,
    ):
        w = width if width is not None else DEFAULT_GRID_SIZE
        h = height if height is not None else DEFAULT_GRID_SIZE
        dc = DISP_CELL

        self.g0 = Geometry(0, 0, dc, dc, w, h)
        self.g1 = Geometry(dc, 0, dc + dc, dc, w, h)
        self.g2 = Geometry(0, dc, dc, dc + dc, dc, dc)
        self.g3 = Geometry(dc, dc, dc + dc, dc + dc, dc, dc)

        self.w, self.h = w, h
        self.size = (w, h)
        self.screen_size = (self.g3.x1, self.g3.y1)
        self.panel_origin = (self.g3.x0, self.g3.y0)

        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Pit Pattern (exp)")
        self.font = pygame.font.Font(None, 28)
        self.clock = pygame.time.Clock()

        self.suite_configs = {cfg.name: cfg for cfg in build_suite("all")}
        self.legacy_presets = LEGACY_PRESETS

        self.eval_settings = eval_settings or EvalRunSettings(
            size=w, seed=DEFAULT_EVAL_RUN.seed, seed_density=DEFAULT_EVAL_RUN.seed_density
        )
        self.eval_repro = eval_repro
        self.active_case_name = "custom"
        self.dynamic_env = False

        self.ui_manager = pygame_gui.UIManager(self.screen_size)
        self._build_ui()

        self.paused = False
        self.step_count = 0
        self.steps_per_frame = 10

        self.grid_mask = np.zeros((h, w), dtype=np.float32)
        self.grid_heterogeneity = np.zeros((h, w), dtype=np.float32)
        self.grid_c = np.zeros((h, w), dtype=np.float32)
        self.grid_h = np.zeros((h, w), dtype=np.float32)
        self.grid_r = np.zeros((h, w), dtype=np.float32)
        self.grid_p_x = np.zeros((h, w), dtype=np.float32)
        self.grid_p_y = np.zeros((h, w), dtype=np.float32)
        self.reset_fields()

    def _build_ui(self) -> None:
        ox, oy = self.panel_origin
        pw = self.g3.gw

        preset_options = ["custom"] + list(self.suite_configs) + [p[0] for p in self.legacy_presets]
        self.preset_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=preset_options,
            starting_option="custom",
            relative_rect=pygame.Rect(ox + 8, oy + 8, pw - 16, 30),
            manager=self.ui_manager,
        )

        btn_y = oy + 46
        btn_w = (pw - 24) // 3
        self.btn_pause = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 8, btn_y, btn_w, 28),
            text="Pause",
            manager=self.ui_manager,
        )
        self.btn_step = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 12 + btn_w, btn_y, btn_w, 28),
            text="Step",
            manager=self.ui_manager,
        )
        self.btn_reset = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 16 + 2 * btn_w, btn_y, btn_w, 28),
            text="Reset",
            manager=self.ui_manager,
        )

        btn_y2 = btn_y + 34
        self.btn_eval_reset = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 8, btn_y2, btn_w, 28),
            text="Eval IC",
            manager=self.ui_manager,
        )
        self.btn_snapshot = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 12 + btn_w, btn_y2, btn_w, 28),
            text="Snapshot",
            manager=self.ui_manager,
        )
        self.btn_eval_mode = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 16 + 2 * btn_w, btn_y2, btn_w, 28),
            text="Eval ON",
            manager=self.ui_manager,
        )

        btn_y3 = btn_y2 + 34
        self.btn_hetero = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 8, btn_y3, btn_w, 28),
            text="Hetero",
            manager=self.ui_manager,
        )
        self.btn_clear_het = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 12 + btn_w, btn_y3, btn_w, 28),
            text="Clear het",
            manager=self.ui_manager,
        )
        self.btn_regen_c = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(ox + 16 + 2 * btn_w, btn_y3, btn_w, 28),
            text="Regen C",
            manager=self.ui_manager,
        )

        scroll_top = btn_y3 + 38
        scroll_h = self.g3.y1 - scroll_top - 8
        self.param_scroll = pygame_gui.elements.UIScrollingContainer(
            relative_rect=pygame.Rect(ox + 8, scroll_top, pw - 16, scroll_h),
            manager=self.ui_manager,
        )

        self.param_labels: dict[str, pygame_gui.elements.UILabel] = {}
        self.param_sliders: dict[str, pygame_gui.elements.UIHorizontalSlider] = {}
        self.param_decimals: dict[str, int] = {}

        y = 4
        slider_w = pw - 48
        for attr, label, minv, maxv, default, decimals in PARAM_SPECS:
            self.param_decimals[attr] = decimals
            self.param_labels[attr] = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(4, y, slider_w, 18),
                text=f"{label}: {default:.{decimals}f}",
                manager=self.ui_manager,
                container=self.param_scroll,
            )
            self.param_sliders[attr] = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(4, y + 20, slider_w, 18),
                start_value=default,
                value_range=(minv, maxv),
                manager=self.ui_manager,
                container=self.param_scroll,
            )
            y += 46

        self.param_scroll.set_scrollable_area_dimensions((slider_w, y + 8))

    def _param(self, attr: str) -> float:
        return float(self.param_sliders[attr].get_current_value())

    def _set_param(self, attr: str, value: float) -> None:
        slider = self.param_sliders[attr]
        lo, hi = slider.value_range
        slider.set_current_value(min(hi, max(lo, value)))
        self._update_param_label(attr)

    def _update_param_label(self, attr: str) -> None:
        label = PARAM_SPECS[[s[0] for s in PARAM_SPECS].index(attr)][1]
        decimals = self.param_decimals[attr]
        value = self._param(attr)
        self.param_labels[attr].set_text(f"{label}: {value:.{decimals}f}")

    def get_model_config(self) -> ModelConfig:
        return ModelConfig(
            name=self.active_case_name,
            description="Live experiment UI",
            feed=self._param("feed"),
            k=self._param("k"),
            d_a=self._param("d_a"),
            d_b=self._param("d_b"),
            reaction_saturation=self._param("reaction_saturation"),
            cubic_damping=self._param("cubic_damping"),
            static_env_scale=self._param("static_env_scale"),
            feed_env_sensitivity=self._param("feed_env_sensitivity"),
            k_env_sensitivity=self._param("k_env_sensitivity"),
            dynamic_env=self.dynamic_env,
            env_rate=self._param("env_rate"),
            env_diffusion=self._param("env_diffusion"),
            env_source=self._param("env_source"),
            env_decay=self._param("env_decay"),
            a_depletion=self._param("a_depletion"),
            uniform_core_decay=self._param("uniform_core_decay"),
            inhibitor_strength=self._param("inhibitor_strength"),
            inhibitor_diffusion=self._param("inhibitor_diffusion"),
            inhibitor_source=self._param("inhibitor_source"),
            inhibitor_decay=self._param("inhibitor_decay"),
        )

    def apply_model_config(self, config: ModelConfig) -> None:
        mapping = {
            "d_a": config.d_a,
            "d_b": config.d_b,
            "feed": config.feed,
            "k": config.k,
            "reaction_saturation": config.reaction_saturation,
            "cubic_damping": config.cubic_damping,
            "a_depletion": config.a_depletion,
            "uniform_core_decay": config.uniform_core_decay,
            "static_env_scale": config.static_env_scale,
            "feed_env_sensitivity": config.feed_env_sensitivity,
            "k_env_sensitivity": config.k_env_sensitivity,
            "env_rate": config.env_rate,
            "env_diffusion": config.env_diffusion,
            "env_source": config.env_source,
            "env_decay": config.env_decay,
            "inhibitor_strength": config.inhibitor_strength,
            "inhibitor_diffusion": config.inhibitor_diffusion,
            "inhibitor_source": config.inhibitor_source,
            "inhibitor_decay": config.inhibitor_decay,
        }
        for attr, value in mapping.items():
            self._set_param(attr, value)

        self.dynamic_env = config.dynamic_env
        self._refresh_eval_mode_button()
        self.regenerate_environment()

    def apply_eval_case(
        self,
        config: ModelConfig,
        settings: EvalRunSettings | None = None,
        *,
        reset: bool = True,
    ) -> None:
        """Apply parameters and ICs matching evaluate_pit_steps."""
        self.eval_settings = settings or EvalRunSettings(
            size=self.w,
            seed=self.eval_settings.seed,
            seed_density=self.eval_settings.seed_density,
        )
        self.eval_repro = True
        self.active_case_name = config.name
        self.apply_model_config(config)
        self._set_param("spatial_feed_delta", 0.0)
        self._set_param("spatial_k_delta", 0.0)
        self.preset_dropdown.selected_option = config.name
        self._refresh_eval_mode_button()
        if reset:
            self.reset_fields()

    def apply_legacy_preset(self, index: int) -> None:
        _, feed, k = self.legacy_presets[index]
        self.eval_repro = False
        self.active_case_name = "custom"
        self._set_param("feed", feed)
        self._set_param("k", k)
        self.preset_dropdown.selected_option = self.legacy_presets[index][0]
        self._refresh_eval_mode_button()

    def _refresh_eval_mode_button(self) -> None:
        self.btn_eval_mode.set_text("Eval ON" if self.eval_repro else "Eval OFF")

    def regenerate_environment(self) -> None:
        config = self.get_model_config()
        self.grid_c = build_static_environment(
            self.h, self.eval_settings.seed, config.static_env_scale
        )

    def reset_fields(self, seed: int | None = None, seed_density: float | None = None) -> None:
        if seed is not None:
            self.eval_settings = EvalRunSettings(
                size=self.eval_settings.size,
                steps=self.eval_settings.steps,
                seed=seed,
                seed_density=self.eval_settings.seed_density,
            )
        if seed_density is not None:
            self.eval_settings = EvalRunSettings(
                size=self.eval_settings.size,
                steps=self.eval_settings.steps,
                seed=self.eval_settings.seed,
                seed_density=seed_density,
            )

        h, w = self.h, self.w
        self.grid_mask = np.zeros((h, w), dtype=np.float32)
        self.grid_heterogeneity = np.zeros((h, w), dtype=np.float32)
        self.grid_h = np.zeros((h, w), dtype=np.float32)
        self.grid_r = np.zeros((h, w), dtype=np.float32)
        config = self.get_model_config()
        if uses_polarity(config):
            self.grid_p_x, self.grid_p_y = make_initial_polarity(
                h, self.eval_settings.seed
            )
        else:
            self.grid_p_x.fill(0.0)
            self.grid_p_y.fill(0.0)

        if self.eval_repro:
            self.grid_a, self.grid_b = make_initial_state(
                h, self.eval_settings.seed, self.eval_settings.seed_density
            )
        else:
            self.grid_a = np.ones((h, w), dtype=np.float32)
            self.grid_b = np.zeros((h, w), dtype=np.float32)
            rng = np.random.default_rng(self.eval_settings.seed)
            cy, cx = h // 2, w // 2
            self.grid_a[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.5
            self.grid_b[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.25
            noise = rng.normal(0, 0.02, (h, w)).astype(np.float32)
            self.grid_a = np.clip(self.grid_a + noise, 0, 1)
            self.grid_b = np.clip(self.grid_b + noise, 0, 1)

        self.regenerate_environment()
        self.step_count = 0

    def random_heterogeneity(self, amount=0.6, sigma=18):
        if self.eval_repro:
            return
        rng = np.random.default_rng()
        noise = rng.uniform(0, 1, (self.h, self.w)).astype(np.float64)
        ksize = max(3, int(2 * sigma) | 1)
        blurred = cv2.GaussianBlur(noise, (ksize, ksize), sigma)
        self.grid_heterogeneity = np.clip(blurred * amount, 0, 1).astype(np.float32)

    def clear_heterogeneity(self):
        self.grid_heterogeneity = np.zeros((self.h, self.w), dtype=np.float32)

    def calc_step(self):
        config = self.get_model_config()
        if self.eval_repro:
            self.grid_a, self.grid_b, self.grid_c, self.grid_h, self.grid_r, self.grid_p_x, self.grid_p_y = calc_step(
                self.grid_a,
                self.grid_b,
                self.grid_c,
                self.grid_h,
                self.grid_r,
                self.grid_p_x,
                self.grid_p_y,
                config,
            )
        else:
            (
                self.grid_a,
                self.grid_b,
                self.grid_c,
                self.grid_h,
                self.grid_r,
                self.grid_p_x,
                self.grid_p_y,
            ) = self._calc_step_with_spatial(config)
        self.step_count += 1

    def _calc_step_with_spatial(self, config: ModelConfig):
        feed, k = effective_parameters(config, self.grid_c)
        blend = np.maximum(self.grid_mask, self.grid_heterogeneity)
        feed = feed + blend * self._param("spatial_feed_delta")
        k = k + blend * self._param("spatial_k_delta")
        return advance_reaction_diffusion(
            self.grid_a,
            self.grid_b,
            self.grid_c,
            self.grid_h,
            self.grid_r,
            self.grid_p_x,
            self.grid_p_y,
            config,
            feed,
            k,
        )

    def save_snapshot(self) -> Path:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = SNAPSHOT_DIR / f"snapshot_{stamp}.png"
        pygame.image.save(self.screen, path)
        return path

    def _on_pause(self) -> None:
        self.paused = not self.paused
        self.btn_pause.set_text("Run" if self.paused else "Pause")

    def _on_eval_mode_toggle(self) -> None:
        self.eval_repro = not self.eval_repro
        self._refresh_eval_mode_button()

    def _handle_ui_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            actions = {
                self.btn_pause: self._on_pause,
                self.btn_step: self.calc_step,
                self.btn_reset: self.reset_fields,
                self.btn_eval_reset: self.reset_fields,
                self.btn_snapshot: lambda: print(f"saved snapshot: {self.save_snapshot()}"),
                self.btn_eval_mode: self._on_eval_mode_toggle,
                self.btn_hetero: self.random_heterogeneity,
                self.btn_clear_het: self.clear_heterogeneity,
                self.btn_regen_c: self.regenerate_environment,
            }
            action = actions.get(event.ui_element)
            if action is not None:
                action()

        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element != self.preset_dropdown:
                return
            selected = event.text
            if selected == "custom":
                self.active_case_name = "custom"
                return
            if selected in self.suite_configs:
                self.apply_eval_case(self.suite_configs[selected])
            elif selected in {p[0] for p in self.legacy_presets}:
                idx = [p[0] for p in self.legacy_presets].index(selected)
                self.apply_legacy_preset(idx)
                self.reset_fields()

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            self.active_case_name = "custom"
            self.preset_dropdown.selected_option = "custom"
            for attr, slider in self.param_sliders.items():
                if event.ui_element == slider:
                    self._update_param_label(attr)
                    if attr == "static_env_scale":
                        self.regenerate_environment()
                    break

    def _handle_keyboard(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.reset_fields()
        elif event.key == pygame.K_SPACE:
            self._on_pause()
        elif event.key == pygame.K_s:
            self.calc_step()
        elif event.key == pygame.K_h:
            self.random_heterogeneity()
        elif event.key == pygame.K_c:
            self.clear_heterogeneity()
        elif event.key == pygame.K_e:
            self.reset_fields()
        elif event.key == pygame.K_1:
            self.apply_legacy_preset(0)
            self.reset_fields()
        elif event.key == pygame.K_2:
            self.apply_legacy_preset(1)
            self.reset_fields()
        elif event.key == pygame.K_3:
            self.apply_legacy_preset(2)
            self.reset_fields()

    def _paint_mouse(self) -> None:
        if self.ui_manager.get_hovering_any_element():
            return

        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if not (click[0] or click[2]):
            return

        x, y = mouse
        if self.g0.within_global(x, y):
            pos = self.g0.get_local(x, y)
            if click[0]:
                cv2.circle(self.grid_a, pos, 10, 0, -5)
                cv2.circle(self.grid_b, pos, 10, 1, -5)
            if click[2]:
                cv2.circle(self.grid_a, pos, 10, 1, -5)
                cv2.circle(self.grid_b, pos, 10, 0, -5)

        if self.g1.within_global(x, y) and not self.eval_repro:
            pos = self.g1.get_local(x, y)
            if click[0]:
                cv2.circle(self.grid_mask, pos, 10, 1, -5)
            if click[2]:
                cv2.circle(self.grid_mask, pos, 10, 0, -5)

    def _render_simulation(self) -> None:
        h, w = self.h, self.w
        self.screen.fill((0, 0, 0))

        img = np.zeros((h, w, 3), dtype=np.int8)
        img[:, :, 0] = np.clip(self.grid_a * 255, 0, 255)
        img[:, :, 1] = 10
        img[:, :, 2] = np.clip(self.grid_b * 255, 0, 255)
        self.g0.blit(self.screen, img)

        c_vis = np.clip((self.grid_c + 1.5) / 3.0 * 255, 0, 255).astype(np.int8)
        img = np.zeros((h, w, 3), dtype=np.int8)
        img[:, :, 0] = np.clip(self.grid_mask * 255, 0, 255)
        img[:, :, 1] = np.clip(self.grid_heterogeneity * 255, 0, 255)
        img[:, :, 2] = c_vis
        self.g1.blit(self.screen, img)
        mask_hint = "eval: mask off" if self.eval_repro else "R=mask G=hetero B=env C"
        self.g1.text(self.screen, mask_hint, (5, 5), self.font)

        config = self.get_model_config()
        status = (
            f"FPS {self.clock.get_fps():.1f} | steps {self.step_count}\n"
            f"case {self.active_case_name} | eval {'ON' if self.eval_repro else 'OFF'}\n"
            f"seed {self.eval_settings.seed} density {self.eval_settings.seed_density:.3f} "
            f"grid {self.w}x{self.h}\n"
            f"feed {config.feed:.4f} k {config.k:.4f} sat {config.reaction_saturation:.2f}\n"
            f"cubic {config.cubic_damping:.4f} dyn {config.dynamic_env} "
            f"env {config.static_env_scale:.2f}\n"
            f"E eval IC  Space pause  S step  Esc reset"
        )
        self.g2.text(self.screen, status, (8, 8), self.font)

    def draw(self, time_delta: float) -> None:
        self._paint_mouse()
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.calc_step()
        self._render_simulation()
        self.ui_manager.update(time_delta)
        self.ui_manager.draw_ui(self.screen)
        pygame.display.update()

    def start(self) -> None:
        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                self.ui_manager.process_events(event)
                self._handle_ui_event(event)
                self._handle_keyboard(event)
            self.draw(time_delta)
        pygame.quit()


def resolve_startup_case(
    args: argparse.Namespace,
) -> tuple[ModelConfig | None, EvalRunSettings, bool]:
    """Return optional config, run settings, and whether to use eval IC."""
    settings = EvalRunSettings(
        size=args.size,
        seed=args.seed,
        seed_density=args.seed_density,
    )
    eval_repro = not args.no_eval_ic

    if args.metrics is not None:
        if not args.case:
            raise SystemExit("--case is required when using --metrics")
        config, metrics_settings = load_eval_case(args.metrics, args.case)
        settings = EvalRunSettings(
            size=args.size if args.size != DEFAULT_GRID_SIZE else metrics_settings.size,
            steps=metrics_settings.steps,
            seed=args.seed if args.seed != DEFAULT_EVAL_RUN.seed else metrics_settings.seed,
            seed_density=(
                args.seed_density
                if args.seed_density != DEFAULT_EVAL_RUN.seed_density
                else metrics_settings.seed_density
            ),
        )
        return config, settings, eval_repro

    if args.preset:
        configs = {cfg.name: cfg for cfg in build_suite("all")}
        if args.preset not in configs:
            raise SystemExit(f"unknown preset {args.preset!r}")
        return configs[args.preset], settings, eval_repro

    return None, settings, eval_repro


def main(argv: list[str] | None = None) -> None:
    args = parse_exp_args(argv)
    config, settings, eval_repro = resolve_startup_case(args)

    if settings.size != args.size and args.metrics is None and args.preset is None:
        print(
            f"note: grid size {args.size}; evaluate_pit_steps default is {DEFAULT_EVAL_RUN.size}",
            flush=True,
        )

    app = DiffusionReaction(
        width=settings.size,
        height=settings.size,
        eval_settings=settings,
        eval_repro=eval_repro,
    )
    if config is not None:
        app.apply_eval_case(config, settings)
    elif eval_repro:
        app.reset_fields()

    app.start()


if __name__ == "__main__":
    main()
