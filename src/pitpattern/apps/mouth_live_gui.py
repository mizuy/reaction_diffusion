#!/usr/bin/env python3
"""Real-time viewer for the crypt-mouth model (pygame).

One state evolves on screen.  Move the p slider and the same crypts and the same field follow:
dots stretch into tubes, open into Ys, grow arms that join into gyri, and the ridges break into
islands, Round (I) -> Elongated (IIIL) -> Branch (IV_B) -> Gyrus -> Villous (IV_V).  The window
looks at a growing surface, so crypts drift outwards as p rises and come back as it falls.

Controls
  p slider / Left, Right   requested progression; the model's p follows at the chosen speed
  speed slider             how fast p follows (per second)
  Play / Space             sweep p up to 4 and back to 0
  coefficient sliders      change the model while it runs
  view                     fixed-width grooves (the model's rendering) or the raw field u > 0
  click / drag in picture  perturb the field: left button paints groove (purple), right button
                           or Shift + click paints mucosa (red); hold to keep pushing.
                           "brush" switches what the left button paints, [ and ] change its size
  Reseed / Reset           new random start / default coefficients
  Save PNG                 writes the current picture to artifacts/pit_mouth/live/

    uv run python -m pitpattern.apps.mouth_live_gui            # or: make mouth-live
    uv run python -m pitpattern.apps.mouth_live_gui --size 330 --p 0.2
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
import pygame_gui

from pitpattern.evaluation.evaluate_pit_mouth import label, measure
from pitpattern.models.pit_mouth_live import Live, LiveConfig
from pitpattern.models.simulate_pit_mouth import params

PANEL_W = 340
STAGE_NAMES = {"Round": "Round (I)", "Elongated": "Elongated (IIIL)", "Branch": "Branch (IV_B)", "Gyrus": "Gyrus (IV)",
               "Villous": "Villous (IV_V)", "transition": "in between"}
# (Config field, label, range, click step)
KNOBS = (
    ("kappa_gyrus", "groove pinning in Gyrus", (0.0, 1.5), 0.05),
    ("g_low", "asymmetry up to Gyrus", (0.0, 1.2), 0.05),
    ("g_villous", "asymmetry in Villous", (-1.6, 0.0), 0.05),
    ("lam_gyrus", "groove pitch in Gyrus (x D)", (1.0, 1.6), 0.05),
    ("lam_villous", "groove pitch in Villous (x D)", (1.3, 2.4), 0.05),
    ("extra", "arm growth budget (px)", (0.0, 1000.0), 25.0),
)


class App:
    def __init__(self, cfg: LiveConfig, scale: int, seed: int, p: float):
        self.cfg0, self.cfg, self.scale, self.seed = cfg, cfg, scale, seed
        self.fixed_width, self.playing, self.play_dir = True, False, 1
        self.brush_groove, self.brush_r = True, 14.0
        self.view = cfg.size * scale
        pygame.init()
        pygame.display.set_caption("Crypt-mouth model (real time)")
        self.screen = pygame.display.set_mode((self.view + PANEL_W, max(self.view, 720)))
        self.ui = pygame_gui.UIManager(self.screen.get_size())
        self.ui.preload_fonts([{"name": "noto_sans", "point_size": 14, "style": "bold", "antialiased": "1"},
                               {"name": "noto_sans", "point_size": 10, "style": "regular", "antialiased": "1"}])
        self.live = Live(cfg, seed, p)
        self.target = float(p)
        self.stage, self.stage_job = "", None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.fps = 0.0
        self._build_panel()

    # ------------------------------------------------------------ UI
    def _build_panel(self):
        x0, w = self.view + 15, PANEL_W - 30
        ui, R = self.ui, pygame.Rect
        E = pygame_gui.elements
        y = 10
        self.lbl_p = E.UILabel(R(x0, y, w, 24), "", ui)
        y += 26
        self.sl_p = E.UIHorizontalSlider(R(x0, y, w, 28), self.target, (0.0, 4.0), ui, click_increment=0.1)
        y += 34
        self.lbl_speed = E.UILabel(R(x0, y, w, 20), "", ui)
        self.sl_speed = E.UIHorizontalSlider(R(x0, y + 20, w, 22), self.cfg.p_rate, (0.02, 0.6), ui, click_increment=0.02)
        y += 50
        half = (w - 10) // 2
        self.btn_play = E.UIButton(R(x0, y, half, 30), "Play", ui)
        self.btn_view = E.UIButton(R(x0 + half + 10, y, half, 30), "view: fixed width", ui)
        y += 38
        self.knobs = []
        for name, text, rng, inc in KNOBS:
            lbl = E.UILabel(R(x0, y, w, 20), "", ui)
            sl = E.UIHorizontalSlider(R(x0, y + 20, w, 22), float(getattr(self.live.m, name)), rng, ui, click_increment=inc)
            self.knobs.append((name, text, lbl, sl))
            y += 46
        third = (w - 20) // 3
        self.btn_reseed = E.UIButton(R(x0, y, third, 30), "Reseed", ui)
        self.btn_reset = E.UIButton(R(x0 + third + 10, y, third, 30), "Reset", ui)
        self.btn_save = E.UIButton(R(x0 + 2 * third + 20, y, third, 30), "Save PNG", ui)
        y += 36
        self.btn_brush = E.UIButton(R(x0, y, w, 28), "", ui)
        y += 34
        self.info = E.UITextBox("", R(x0, y, w, self.screen.get_height() - y - 10), ui)
        self._last_info = None

    def _refresh_labels(self):
        L = self.live
        moving = "" if abs(L.p - self.target) < 1e-9 else f"   -> {self.target:.2f}"
        self.lbl_p.set_text(f"progression p = {L.p:.2f}{moving}")
        self.lbl_speed.set_text(f"speed = {L.cfg.p_rate:.2f} per second")
        brush = f"click paints: {'groove (purple)' if self.brush_groove else 'mucosa (red)'}, radius {self.brush_r:.0f} px"
        if self.btn_brush.text != brush:
            self.btn_brush.set_text(brush)
        for name, text, lbl, _ in self.knobs:
            lbl.set_text(f"{text} = {getattr(L.m, name):.2f}")
        pr = params(L.p, L.m)
        text = (f"<b>{STAGE_NAMES.get(self.stage, self.stage or '...')}</b><br>"
                f"crypts in view {L.n_visible()}   scale x{L.scale(L.p):.2f}<br>"
                f"{self.fps:.0f} frames per second<br><br>"
                f"<font size=2>coefficients at p = {L.p:.2f}<br>"
                f"unit pinning {float(pr['kappa']):.2f}   groove pinning {float(pr['kappa_g']):.2f}<br>"
                f"pitch {float(pr['lam']) / L.m.D:.2f} D   asymmetry {float(pr['g']):+.2f}<br>"
                f"arm length target {float(pr['grow']) * L.m.extra / 3:.0f} px</font>")
        if text != self._last_info:
            self.info.set_text(text)
            self._last_info = text

    def set_target(self, p: float, from_slider: bool = False):
        self.target = float(np.clip(p, 0.0, 4.0))
        self.live.p_target = self.target
        if not from_slider:
            self.sl_p.set_current_value(self.target)

    def handle(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.set_target(self.target + (0.1 if event.key == pygame.K_RIGHT else -0.1))
            elif event.key == pygame.K_SPACE:
                self.toggle_play()
            elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                self.brush_r = float(np.clip(self.brush_r + (3 if event.key == pygame.K_RIGHTBRACKET else -3), 5, 60))
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element is self.sl_p:
                self.set_target(event.value, from_slider=True)
            elif event.ui_element is self.sl_speed:
                self.live.cfg = replace(self.live.cfg, p_rate=float(event.value))
            for name, _, _, sl in self.knobs:
                if event.ui_element is sl:
                    step = next(k[3] for k in KNOBS if k[0] == name)
                    self.live.m = replace(self.live.m, **{name: round(float(event.value) / step) * step})
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element is self.btn_play:
                self.toggle_play()
            elif event.ui_element is self.btn_view:
                self.fixed_width = not self.fixed_width
                self.btn_view.set_text(f"view: {'fixed width' if self.fixed_width else 'raw field'}")
            elif event.ui_element is self.btn_reseed:
                self.seed += 1
                self.restart(self.live.m)
            elif event.ui_element is self.btn_reset:
                for name, _, _, sl in self.knobs:
                    sl.set_current_value(float(getattr(self.cfg0.base, name)))
                self.live.m = self.cfg0.base
            elif event.ui_element is self.btn_save:
                self.save()
            elif event.ui_element is self.btn_brush:
                self.brush_groove = not self.brush_groove
        self.ui.process_events(event)
        return True

    def restart(self, m):
        p, rate = self.live.p, self.live.cfg.p_rate
        self.live = Live(replace(self.cfg0, base=m, p_rate=rate), self.seed, p)
        self.live.p_target = self.target

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.set_text("Stop" if self.playing else "Play")
        if self.playing:
            self.play_dir = -1 if self.live.p >= 3.99 else 1
            self.set_target(4.0 if self.play_dir > 0 else 0.0)

    def paint(self):
        """While a mouse button is held inside the picture, keep forcing the field under it."""
        left, _, right = pygame.mouse.get_pressed()
        x, y = pygame.mouse.get_pos()
        if not (left or right) or not (0 <= x < self.view and 0 <= y < self.view):
            return
        shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
        groove = self.brush_groove if left and not shift else not self.brush_groove
        self.live.paint(x / self.scale, y / self.scale, groove, self.brush_r)

    def update(self):
        L = self.live
        if self.playing and abs(L.p - self.target) < 1e-9:       # arrived: turn round
            self.play_dir = -self.play_dir
            self.set_target(4.0 if self.play_dir > 0 else 0.0)
        L.step()
        self.paint()                    # after the step, so a held brush stays visible
        if self.stage_job is None or self.stage_job.done():
            if self.stage_job is not None and self.stage_job.exception() is None:
                self.stage = self.stage_job.result()
            binary = L.binary()
            mcfg = replace(L.m, width=L.cfg.size, height=L.cfg.size)
            self.stage_job = self.pool.submit(lambda: label(measure(binary, mcfg, periodic=False)))

    def draw(self):
        self.rgb = self.live.rgb(self.fixed_width)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(self.rgb.transpose(1, 0, 2)))
        surf = pygame.transform.smoothscale(surf, (self.view, self.view))
        self.screen.fill((24, 24, 28))
        self.screen.blit(surf, (0, 0))
        self._refresh_labels()
        self.ui.draw_ui(self.screen)
        pygame.display.flip()

    def save(self, path: Path | None = None):
        from PIL import Image
        out = path or Path("artifacts/pit_mouth/live") / f"live_p{self.live.p:.2f}_seed{self.seed}_{int(time.time())}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgb).save(out)
        print("saved", out)

    def run(self, frames: int | None = None, screenshot: Path | None = None, script=None):
        clock = pygame.time.Clock()
        n = 0
        running = True
        while running:
            dt = clock.tick(30) / 1000.0
            self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-3) if self.fps else 1 / max(dt, 1e-3)
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            if script:
                script(self, n)
            self.ui.update(dt)
            self.update()
            self.draw()
            n += 1
            if frames is not None and n >= frames:
                break
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.screen, str(screenshot))
        self.pool.shutdown(wait=False, cancel_futures=True)
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=330, help="side of the window (px); 330 gives a fast 432 px grid")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--speed", type=float, default=LiveConfig().p_rate)
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames (testing)")
    ap.add_argument("--screenshot", type=Path, default=None)
    a = ap.parse_args()
    App(replace(LiveConfig(), size=a.size, p_rate=a.speed), a.scale, a.seed, a.p).run(a.frames, a.screenshot)


if __name__ == "__main__":
    main()
