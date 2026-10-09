#!/usr/bin/env python3
"""Interactive viewer for the crypt lobulation model (pygame).

Move the p slider and watch the lobes regroup: dots (I) -> straight tubes
(IIIL) -> Ys (IV_B) -> network with red islands (IV_V).  Each change of p
starts a short re-anneal that runs a few sweeps per frame, so the pattern
morphs continuously instead of jumping.

Controls
  p slider            progression (all model energies are functions of p only)
  gradient            on: p varies left -> right around the slider value (slide-6 style)
  spread slider       range of p across the field when the gradient is on
  temperature slider  temperature of the live dynamics between re-anneals (0 = frozen)
  Reseed              new random lattice and random initial state
  Anneal              full anneal at the current p (as in the validation runs)
  Save PNG            writes the current picture to artifacts/pit_crypt/gui/

    uv run python -m pitpattern.apps.crypt_gui            # or: make crypt-gui
    uv run python -m pitpattern.apps.crypt_gui --cols 18 --rows 16
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
import pygame_gui

from pitpattern.models.simulate_pit_crypt import (
    Config, CryptModel, Lattice, composition, label, relax, render)

PANEL_W = 330
BAND_NAMES = {"I": "Type I", "IIIL": "Type IIIL", "IV_B": "Type IV_B", "IV_V": "Type IV_V",
              "transition": "transition"}


class App:
    def __init__(self, cfg: Config, scale: int, seed: int, p: float):
        self.cfg, self.scale = cfg, scale
        self.rng = np.random.default_rng(seed)
        self.p, self.gradient, self.spread, self.T_live = p, False, 3.0, 0.03
        self.schedule: list[float] = []          # pending sweep temperatures
        self.W, self.H = (int(round(v)) for v in cfg.box)
        pygame.init()
        pygame.display.set_caption("Crypt lobulation model")
        self.screen = pygame.display.set_mode((self.W * scale + PANEL_W, max(self.H * scale, 560)))
        self.ui = pygame_gui.UIManager(self.screen.get_size())
        self.ui.preload_fonts([{"name": "noto_sans", "point_size": 14, "style": "bold", "antialiased": "1"},
                               {"name": "noto_sans", "point_size": 10, "style": "regular", "antialiased": "1"}])
        self._build_panel()
        self.reseed()

    # ------------------------------------------------------------ model
    def p_field(self):
        if not self.gradient:
            return lambda pos: self.p
        Lx = self.cfg.box[0]
        return lambda pos: self.p + self.spread * (pos[0] / Lx - 0.5)

    def reseed(self):
        cfg = self.cfg
        lat = Lattice(relax(cfg.n_lobes, cfg.box, cfg.spacing, self.rng, cfg.relax_steps), cfg.box, cfg.spacing)
        self.model = CryptModel(lat, self.p_field(), cfg)
        self.model.randomise(self.rng)
        self.full_anneal(sweeps=250)

    def _temps(self, T0: float, n: int) -> list[float]:
        T1 = self.cfg.T1
        return [T0 * (T1 / T0) ** (i / max(n - 1, 1)) for i in range(n)]

    def full_anneal(self, sweeps: int = 500):
        T0 = max(2.0, float(np.abs(self.model.gain).max()))
        self.schedule = self._temps(T0, sweeps)

    def on_p_changed(self):
        self.model.set_p(self.p_field())
        # reheat to a fraction of the largest binding energy: lobes regroup and IV_V domains
        # can still order, while the picture morphs instead of being scrambled
        T0 = max(1.5, 0.5 * float(np.abs(self.model.gain).max()))
        self.schedule = self._temps(T0, 150)

    def step(self, budget_s: float = 0.03):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < budget_s:
            if self.schedule:
                self.model.sweep(self.rng, self.schedule.pop(0))
            elif self.T_live > 0:
                self.model.sweep(self.rng, self.T_live)
            else:
                break

    # ------------------------------------------------------------ UI
    def _build_panel(self):
        x0, w = self.W * self.scale + 15, PANEL_W - 30
        ui, R = self.ui, pygame.Rect
        y = 12
        self.lbl_p = pygame_gui.elements.UILabel(R(x0, y, w, 24), "", ui)
        y += 26
        self.sl_p = pygame_gui.elements.UIHorizontalSlider(R(x0, y, w, 26), self.p, (-0.5, 3.6), ui, click_increment=0.05)
        y += 40
        self.btn_grad = pygame_gui.elements.UIButton(R(x0, y, w, 30), "gradient: off", ui)
        y += 38
        self.lbl_spread = pygame_gui.elements.UILabel(R(x0, y, w, 24), "", ui)
        y += 26
        self.sl_spread = pygame_gui.elements.UIHorizontalSlider(R(x0, y, w, 26), self.spread, (0.0, 4.5), ui, click_increment=0.1)
        y += 40
        self.lbl_T = pygame_gui.elements.UILabel(R(x0, y, w, 24), "", ui)
        y += 26
        self.sl_T = pygame_gui.elements.UIHorizontalSlider(R(x0, y, w, 26), self.T_live, (0.0, 0.5), ui, click_increment=0.01)
        y += 44
        half = (w - 10) // 2
        self.btn_reseed = pygame_gui.elements.UIButton(R(x0, y, half, 32), "Reseed", ui)
        self.btn_anneal = pygame_gui.elements.UIButton(R(x0 + half + 10, y, half, 32), "Anneal", ui)
        y += 40
        self.btn_save = pygame_gui.elements.UIButton(R(x0, y, w, 32), "Save PNG", ui)
        y += 46
        self.info = pygame_gui.elements.UITextBox("", R(x0, y, w, 230), ui)
        self._last_info = None

    def _refresh_labels(self):
        self.lbl_p.set_text(f"progression p = {self.p:.2f}" + ("  (centre)" if self.gradient else ""))
        self.lbl_spread.set_text(f"gradient spread = {self.spread:.1f}")
        self.lbl_T.set_text(f"live temperature = {self.T_live:.2f}")
        c = composition(self.model)
        band = label(c) if not self.gradient else "(gradient)"
        text = (f"<b>{BAND_NAMES.get(band, band)}</b><br>"
                f"lobes {c['n_lobes']}<br>tubes {c['tubes']}   Ys {c['ys']}<br>"
                f"valence 0/1/2/3: {c['v0']} / {c['v1']} / {c['v2']} / {c['v3']}<br>"
                f"{'annealing...' if self.schedule else 'settled'}<br><br>"
                "<font size=2>I: p &lt; 0.5 &nbsp; IIIL: 0.6-1.2<br>IV_B: 2.0-2.6 &nbsp; IV_V: p &gt; 2.8</font>")
        if text != self._last_info:
            self.info.set_text(text)
            self._last_info = text

    def handle(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element is self.sl_p:
                self.p = float(event.value)
                self.on_p_changed()
            elif event.ui_element is self.sl_spread:
                self.spread = float(event.value)
                if self.gradient:
                    self.on_p_changed()
            elif event.ui_element is self.sl_T:
                self.T_live = float(event.value)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element is self.btn_reseed:
                self.reseed()
            elif event.ui_element is self.btn_anneal:
                self.full_anneal()
            elif event.ui_element is self.btn_grad:
                self.gradient = not self.gradient
                self.btn_grad.set_text(f"gradient: {'on' if self.gradient else 'off'}")
                self.on_p_changed()
            elif event.ui_element is self.btn_save:
                self.save()
        self.ui.process_events(event)
        return True

    def draw(self):
        rgb, _ = render(self.model.lat, self.model.tubes, self.model.ys, self.cfg, ss=2)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
        surf = pygame.transform.smoothscale(surf, (self.W * self.scale, self.H * self.scale))
        self.screen.fill((24, 24, 28))
        self.screen.blit(surf, (0, 0))
        self._refresh_labels()
        self.ui.draw_ui(self.screen)
        pygame.display.flip()
        self.last_rgb = rgb

    def save(self, path: Path | None = None):
        from PIL import Image
        out = path or Path("artifacts/pit_crypt/gui") / f"crypt_p{self.p:.2f}{'_grad' if self.gradient else ''}_{int(time.time())}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.last_rgb).save(out)
        print("saved", out)

    def run(self, frames: int | None = None, screenshot: Path | None = None, script=None):
        clock = pygame.time.Clock()
        n = 0
        running = True
        while running:
            dt = clock.tick(30) / 1000.0
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            if script:
                script(self, n)
            self.ui.update(dt)
            self.step()
            self.draw()
            n += 1
            if frames is not None and n >= frames:
                break
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.screen, str(screenshot))
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=14)
    ap.add_argument("--rows", type=int, default=14)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--frames", type=int, default=None, help="stop after N frames (testing)")
    ap.add_argument("--screenshot", type=Path, default=None)
    a = ap.parse_args()
    App(replace(Config(), cols=a.cols, rows=a.rows), a.scale, a.seed, a.p).run(a.frames, a.screenshot)


if __name__ == "__main__":
    main()
