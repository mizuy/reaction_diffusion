#!/usr/bin/env python3
"""Interactive viewer for the crypt-mouth model (pygame).

Drag the p slider from 0 to 4 and the pattern goes Round (I) -> Elongated (IIIL) ->
Branch (IV_B) -> Gyrus -> Villous (IV_V).  Every picture is an independent run of the model at
that p (random start, same seed), computed by background workers and cached; on start the whole
p range is filled in, nearest values first, so the slider soon scrubs without waiting.  While a
picture is being computed the nearest finished one is shown.

Controls
  p slider / Left, Right   progression (every model coefficient is a function of p only)
  Play / Space             run p from 0 to 4 through the finished pictures
  gradient                 one wide field with p rising 0 -> 4 from left to right
  coefficient sliders      change the model (the cache is refilled for the new values)
  Reseed                   new random start
  Reset                    back to the validated coefficients
  Save PNG                 writes the current picture to artifacts/pit_mouth/gui/

    uv run python -m pitpattern.apps.mouth_gui            # or: make mouth-gui
    uv run python -m pitpattern.apps.mouth_gui --size 440 --workers 10
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
import pygame_gui

from pitpattern.models.simulate_pit_mouth import Config, params

PANEL_W = 340
STAGE_NAMES = {"Round": "Round (I)", "Elongated": "Elongated (IIIL)", "Branch": "Branch (IV_B)", "Gyrus": "Gyrus (IV)",
               "Villous": "Villous (IV_V)", "transition": "transition", "gradient": "gradient 0 -> 4"}
# (Config field, label, range, click step)
KNOBS = (
    ("kappa_gyrus", "groove pinning in Gyrus", (0.0, 1.5), 0.05),
    ("g_low", "asymmetry up to Gyrus", (0.0, 1.2), 0.05),
    ("g_villous", "asymmetry in Villous", (-1.6, 0.0), 0.05),
    ("lam_gyrus", "groove pitch in Gyrus (x D)", (1.0, 1.6), 0.05),
    ("lam_villous", "groove pitch in Villous (x D)", (1.3, 2.4), 0.05),
    ("extra", "arm growth budget (px)", (0.0, 800.0), 20.0),
)
GRID = tuple(round(0.1 * i, 2) for i in range(41))


def compute(p: float | None, seed: int, cfg: Config):
    """One independent run (worker process).  p = None: the left-to-right gradient."""
    from pitpattern.evaluation.evaluate_pit_mouth import label, measure
    from pitpattern.models.simulate_pit_mouth import render, run

    if p is None:
        pad = 40
        cfg = replace(cfg, width=3 * cfg.width + 2 * pad)
        px = np.interp((np.arange(cfg.width) - pad) / (cfg.width - 2 * pad),
                       [0.0, 0.03, 0.20, 0.23, 0.31, 0.43, 0.57, 0.73, 0.92, 1.0], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        states, info = run(px, seed, cfg)
        rgb, _ = render(states[-1][1], cfg)
        return rgb[:, pad:-pad], "gradient", info["n_crypts"]
    states, info = run(p, seed, cfg)
    rgb, binary = render(states[-1][1], cfg)
    return rgb, label(measure(binary, cfg)), info["n_crypts"]


class App:
    def __init__(self, cfg: Config, scale: int, seed: int, p: float, workers: int):
        self.base, self.cfg, self.scale, self.seed = cfg, cfg, scale, seed
        self.p, self.gradient, self.playing = round(p, 2), False, False
        self.cache: dict = {}            # (p or None, seed, cfg) -> (rgb, label, n_crypts)
        self.jobs: dict = {}             # same key -> Future
        self.pool = ProcessPoolExecutor(max_workers=workers)
        self.dirty_at: float | None = None
        self.view = cfg.width * scale
        pygame.init()
        pygame.display.set_caption("Crypt-mouth model")
        self.screen = pygame.display.set_mode((self.view + PANEL_W, max(self.view, 700)))
        self.ui = pygame_gui.UIManager(self.screen.get_size())
        self.ui.preload_fonts([{"name": "noto_sans", "point_size": 14, "style": "bold", "antialiased": "1"},
                               {"name": "noto_sans", "point_size": 10, "style": "regular", "antialiased": "1"}])
        self._build_panel()
        self.shown = None
        self.refill()

    # ------------------------------------------------------------ jobs
    def key(self, p):
        return (p, self.seed, self.cfg)

    def request(self, p):
        k = self.key(p)
        if k not in self.cache and k not in self.jobs:
            self.jobs[k] = self.pool.submit(compute, p, self.seed, self.cfg)

    def refill(self):
        """Drop jobs for other settings, then queue the current picture and the whole p range."""
        for k, f in list(self.jobs.items()):
            if k[1:] != (self.seed, self.cfg) and f.cancel():
                del self.jobs[k]
        self.request(None if self.gradient else self.p)
        for p in sorted(GRID, key=lambda v: abs(v - self.p)):
            self.request(p)

    def collect(self):
        for k, f in list(self.jobs.items()):
            if f.done():
                del self.jobs[k]
                if not f.cancelled() and f.exception() is None:
                    self.cache[k] = f.result()
                elif not f.cancelled():
                    print("worker failed:", f.exception())

    def current(self):
        """(shown p, frame) for the slider value: the exact picture, or the nearest finished one."""
        if self.gradient:
            return None, self.cache.get(self.key(None))
        if self.key(self.p) in self.cache:
            return self.p, self.cache[self.key(self.p)]
        near = [k[0] for k in self.cache if k[0] is not None and k[1:] == (self.seed, self.cfg)]
        if not near:
            return None, None
        q = min(near, key=lambda v: abs(v - self.p))
        return q, self.cache[self.key(q)]

    def set_p(self, p: float, from_slider: bool = False):
        self.p = round(float(np.clip(p, 0, 4)) / 0.05) * 0.05
        self.p = round(self.p, 2)
        if not from_slider:
            self.sl_p.set_current_value(self.p)
        self.request(self.p)

    # ------------------------------------------------------------ UI
    def _build_panel(self):
        x0, w = self.view + 15, PANEL_W - 30
        ui, R = self.ui, pygame.Rect
        E = pygame_gui.elements
        y = 10
        self.lbl_p = E.UILabel(R(x0, y, w, 24), "", ui)
        y += 26
        self.sl_p = E.UIHorizontalSlider(R(x0, y, w, 28), self.p, (0.0, 4.0), ui, click_increment=0.1)
        y += 36
        half = (w - 10) // 2
        self.btn_play = E.UIButton(R(x0, y, half, 30), "Play", ui)
        self.btn_grad = E.UIButton(R(x0 + half + 10, y, half, 30), "gradient: off", ui)
        y += 40
        self.knobs = []
        for name, text, rng, inc in KNOBS:
            lbl = E.UILabel(R(x0, y, w, 20), "", ui)
            sl = E.UIHorizontalSlider(R(x0, y + 20, w, 22), float(getattr(self.cfg, name)), rng, ui, click_increment=inc)
            self.knobs.append((name, text, lbl, sl))
            y += 48
        third = (w - 20) // 3
        self.btn_reseed = E.UIButton(R(x0, y, third, 30), "Reseed", ui)
        self.btn_reset = E.UIButton(R(x0 + third + 10, y, third, 30), "Reset", ui)
        self.btn_save = E.UIButton(R(x0 + 2 * third + 20, y, third, 30), "Save PNG", ui)
        y += 40
        self.info = E.UITextBox("", R(x0, y, w, self.screen.get_height() - y - 10), ui)
        self._last_info = None

    def _refresh_labels(self, shown_p, frame):
        self.lbl_p.set_text(f"progression p = {self.p:.2f}")
        for name, text, lbl, _ in self.knobs:
            lbl.set_text(f"{text} = {getattr(self.cfg, name):.2f}")
        total = len(GRID)
        ready = sum(self.key(p) in self.cache for p in GRID)
        pr = params(self.p, self.cfg)
        if frame is None:
            head = "<b>computing...</b>"
        else:
            _, lab, n = frame
            wait = "" if self.gradient or shown_p == self.p else f"  (showing p = {shown_p:.2f}, computing)"
            head = f"<b>{STAGE_NAMES.get(lab, lab)}</b>{wait}<br>crypts {n}"
        text = (f"{head}<br>seed {self.seed}   pictures ready {ready} / {total}<br><br>"
                f"<font size=2>coefficients at p = {self.p:.2f}<br>"
                f"unit pinning {float(pr['kappa']):.2f}   groove pinning {float(pr['kappa_g']):.2f}<br>"
                f"pitch {float(pr['lam']) / self.cfg.D:.2f} D   asymmetry {float(pr['g']):+.2f}<br>"
                f"area per crypt x{float(pr['expand']):.2f}<br><br>"
                "Round p &lt; 0.5 &nbsp; Elongated 0.6-1.4<br>Branch 1.5-2.5 &nbsp; Gyrus 2.9-3.2<br>Villous p &gt; 3.5"
                "<br>(ranges of the validated coefficients)</font>")
        if text != self._last_info:
            self.info.set_text(text)
            self._last_info = text

    def handle(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.set_p(self.p + (0.1 if event.key == pygame.K_RIGHT else -0.1))
            elif event.key == pygame.K_SPACE:
                self.toggle_play()
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element is self.sl_p:
                self.set_p(event.value, from_slider=True)
            for name, _, _, sl in self.knobs:
                if event.ui_element is sl:
                    step = next(k[3] for k in KNOBS if k[0] == name)
                    self.cfg = replace(self.cfg, **{name: round(float(event.value) / step) * step})
                    self.dirty_at = time.perf_counter()      # refill once the slider rests
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element is self.btn_play:
                self.toggle_play()
            elif event.ui_element is self.btn_grad:
                self.gradient = not self.gradient
                self.btn_grad.set_text(f"gradient: {'on' if self.gradient else 'off'}")
                self.request(None if self.gradient else self.p)
            elif event.ui_element is self.btn_reseed:
                self.seed += 1
                self.refill()
            elif event.ui_element is self.btn_reset:
                self.cfg = self.base
                for name, _, _, sl in self.knobs:
                    sl.set_current_value(float(getattr(self.cfg, name)))
                self.refill()
            elif event.ui_element is self.btn_save:
                self.save()
        self.ui.process_events(event)
        return True

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.set_text("Stop" if self.playing else "Play")
        if self.playing and self.p >= 4.0:
            self.set_p(0.0)
        self._play_t = time.perf_counter()

    def update(self):
        self.collect()
        if self.dirty_at is not None and time.perf_counter() - self.dirty_at > 0.4:
            self.dirty_at = None
            self.refill()
        if self.playing and time.perf_counter() - self._play_t > 0.35:
            self._play_t = time.perf_counter()
            if self.p >= 4.0:
                self.toggle_play()
            elif self.key(round(self.p + 0.1, 2)) in self.cache:      # wait for the next picture
                self.set_p(self.p + 0.1)

    def draw(self):
        shown_p, frame = self.current()
        self.screen.fill((24, 24, 28))
        if frame is not None:
            rgb = frame[0]
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
            h = int(round(self.view * rgb.shape[0] / rgb.shape[1]))
            surf = pygame.transform.smoothscale(surf, (self.view, h))
            self.screen.blit(surf, (0, (self.view - h) // 2))
            self.shown = rgb
        self._refresh_labels(shown_p, frame)
        self.ui.draw_ui(self.screen)
        pygame.display.flip()

    def save(self, path: Path | None = None):
        from PIL import Image
        if self.shown is None:
            return
        tag = "gradient" if self.gradient else f"p{self.p:.2f}"
        out = path or Path("artifacts/pit_mouth/gui") / f"mouth_{tag}_seed{self.seed}_{int(time.time())}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.shown).save(out)
        print("saved", out)

    def run(self, frames: int | None = None, screenshot: Path | None = None, wait_ready: bool = False):
        clock = pygame.time.Clock()
        n = 0
        running = True
        while running:
            dt = clock.tick(30) / 1000.0
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            self.ui.update(dt)
            self.update()
            self.draw()
            n += 1
            shown_p, frame = self.current()
            ready = frame is not None and (self.gradient or shown_p == self.p)
            if frames is not None and n >= frames and (ready or not wait_ready):
                break
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.screen, str(screenshot))
        self.pool.shutdown(wait=False, cancel_futures=True)
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=330, help="side of the periodic box (px); 440 = validation size")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames (testing)")
    ap.add_argument("--screenshot", type=Path, default=None)
    a = ap.parse_args()
    cfg = replace(Config(), width=a.size, height=a.size)
    App(cfg, a.scale, a.seed, a.p, a.workers).run(a.frames, a.screenshot, wait_ready=a.screenshot is not None)


if __name__ == "__main__":
    main()
