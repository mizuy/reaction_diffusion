# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import pygame
from pygame.locals import *

# Modified kernel for shorter patterns
kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])


class Geometry:
    def __init__(self, x0, y0, x1, y1, w, h):
        assert x1 > x0 >= 0
        assert y1 > y0 >= 0
        assert w > 0 and h > 0

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
        return (self.x0 <= gx < self.x1) and (self.y0 <= gy < self.y1)

    def within_local(self, lx, ly):
        return (0 <= lx < self.lw) and (0 <= ly < self.lh)

    def get_global(self, lx, ly):
        return self.x0 + int(lx * self.sw), self.y0 + int(ly * self.sh)

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

    def text(self, screen, text, pos, font, color):
        pos = (self.pos[0] + pos[0], self.pos[1] + pos[1])
        blit_text(screen, text, pos, font, color)


def blit_text(surface, text, pos, font, color=pygame.Color("black")):
    words = [word.split(" ") for word in text.splitlines()]
    space = font.size(" ")[0]
    max_width, max_height = surface.get_size()
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


class Slider:
    def __init__(self, rect):
        assert isinstance(rect, Rect)
        self.rect = rect
        self.slider_rect = rect.copy()
        self.slider_rect.inflate_ip(-20, -20)
        self.knob_rect = rect.copy()
        self.knob_rect.move_ip(10, 0)
        self.knob_rect.width = 4
        self.min_value = 0
        self.max_value = 1
        self._value = 0

    def initialize(self, min_value, max_value, value):
        self.min_value = min_value
        self.max_value = max_value
        self._value = value

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 0, 0), self.rect)
        pygame.draw.rect(surface, (128, 128, 128), self.slider_rect)
        pygame.draw.rect(surface, (0, 0, 255), self.knob_rect)

    def set_pos(self, xpos):
        xpos = max(self.slider_rect.left, min(self.slider_rect.right, xpos))
        ypos = self.knob_rect.center[1]
        self.knob_rect.center = (xpos, ypos)

    @property
    def value(self):
        ratio = (self.knob_rect.center[0] - self.slider_rect.left) / self.slider_rect.width
        return self.min_value + ratio * (self.max_value - self.min_value)

    @value.setter
    def value(self, value):
        value = min(self.max_value, max(self.min_value, value))
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        self.set_pos(self.slider_rect.left + ratio * self.slider_rect.width)

    def on_click(self, pos):
        if self.rect.collidepoint(pos):
            self.set_pos(pos[0])
            return True
        return False


class SliderText:
    def __init__(self, text_pos, slider_rect, color=pygame.Color("white")):
        self.text_pos = text_pos
        self.slider = Slider(slider_rect)
        self.color = color
        self.text_format = ""
        self.text = ""

    def initialize(self, text_format, min_value, max_value, value):
        self.slider.initialize(min_value, max_value, value)
        self.text_format = text_format
        self.value = value

    @property
    def value(self):
        return self.slider.value

    @value.setter
    def value(self, value):
        self.slider.value = value
        self.text = self.text_format.format(value)

    def draw(self, surface, font):
        t = font.render(self.text, 1, self.color)
        surface.blit(t, self.text_pos)
        self.slider.draw(surface)

    def on_click(self, pos):
        s = self.slider.on_click(pos)
        if s:
            self.value = self.slider.value


class DiffusionReaction:
    def __init__(self, width=512, height=256):
        w, h = width, height
        # g0 g1
        # g2 g3
        self.g0 = Geometry(0, 0, w, h, w, h)
        self.g1 = Geometry(w, 0, w + w, h, w, h)
        self.g2 = Geometry(0, h, w, h + h, w, h)
        self.g3 = Geometry(w, h, w + w, h + h, w, h)

        self.w, self.h = w, h
        self.size = (w, h)

        self.screen_size = (self.g3.x1, self.g3.y1)
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Diffusion Reaction - Pit Pattern Simulation")
        self.font = pygame.font.Font(None, 30)
        self.clock = pygame.time.Clock()

        self.delta = 1

        # Optimized parameters for pit patterns
        # Type I (normal): spots
        # Type III (small tubular): shorter patterns
        # Type IV (gyrus-like): branching but compact

        # Initial values - better tuned for pit patterns
        dA_base = 1.0
        dB_base = 0.5
        feed0, k0 = 0.055, 0.062  # Type I - normal round pits
        feed1, k1 = 0.075, 0.056  # Type III/IV - adjusted for shorter patterns

        variables = [
            ("dA0", "dA0:{0:.2f}", 0.5, 2.0, dA_base, "white"),
            ("dB0", "dB0:{0:.2f}", 0.2, 1.0, dB_base, "white"),
            ("dA1", "dA1:{0:.2f}", 0.3, 1.5, 0.7, "white"),
            ("dB1", "dB1:{0:.2f}", 0.3, 1.2, 0.6, "white"),
            ("feed0", "feed0:{0:.3f}", 0.02, 0.08, feed0, "Blue"),
            ("k0", "k0:{0:.3f}", 0.050, 0.070, k0, "Blue"),
            ("feed1", "feed1:{0:.3f}", 0.050, 0.10, feed1, "Red"),
            ("k1", "k1:{0:.3f}", 0.050, 0.065, k1, "Red"),
        ]

        self.sliders = []
        for i, (name, txt, minv, maxv, val, color) in enumerate(variables):
            pad = 3
            th = 15  # text height
            sh = 15  # slider height
            hh = th + 1 + sh + pad

            tx0, ty0 = self.g3.get_global(pad, pad + i * hh)
            rx0, ry0 = self.g3.get_global(pad, pad + i * hh + th + 2)
            rx1, ry1 = self.g3.get_global(self.g3.lw - 2 * pad, pad + i * hh + th + 2 + sh)
            slider = SliderText((tx0, ty0), Rect(rx0, ry0, rx1 - rx0, ry1 - ry0))

            slider.initialize(txt, minv, maxv, val)
            slider.color = pygame.Color(color)

            setattr(self, name, slider)
            self.sliders.append(slider)

        # Create gradient mask (0 to 1 from left to right)
        self.grid_mask = np.linspace(0, 1, w)

        # Initialize grids with better initial conditions
        self.grid_a = np.ones((h, w))
        self.grid_b = np.zeros((h, w))

        # Add initial perturbations for pattern formation
        # Multiple small random seeds for shorter patterns
        np.random.seed(42)
        for _ in range(20):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            r = 3  # Small radius for compact patterns
            y_low, y_high = max(0, y - r), min(h, y + r)
            x_low, x_high = max(0, x - r), min(w, x + r)
            self.grid_b[y_low:y_high, x_low:x_high] = 1

        # Pit pattern presets (tuned for shorter patterns)
        self.presets = [
            (0.055, 0.062),  # Type I - normal round pits
            (0.065, 0.058),  # Type III - small tubular
            (0.075, 0.056),  # Type IV - gyrus-like but compact
        ]

    def reset_values(self):
        w, h = self.w, self.h
        self.grid_a = np.ones((h, w))
        self.grid_b = np.zeros((h, w))
        # Re-add initial perturbations
        for _ in range(20):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            r = 3
            y_low, y_high = max(0, y - r), min(h, y + r)
            x_low, x_high = max(0, x - r), min(w, x + r)
            self.grid_b[y_low:y_high, x_low:x_high] = 1

    def set_preset(self, preset_i):
        feed, k = self.presets[preset_i]
        self.feed0.value = feed
        self.k0.value = k
        # Also adjust diffusion for the preset
        if preset_i == 0:  # Type I
            self.dA0.value = 1.0
            self.dB0.value = 0.5
        elif preset_i == 1:  # Type III
            self.dA0.value = 0.8
            self.dB0.value = 0.6
        elif preset_i == 2:  # Type IV
            self.dA0.value = 0.7
            self.dB0.value = 0.55

    def calc_step(self):
        a = self.grid_a
        b = self.grid_b
        ab2 = a * (b**2)

        # Spatially varying parameters
        feed = np.interp(self.grid_mask, [0, 1], [self.feed0.value, self.feed1.value])
        k = np.interp(self.grid_mask, [0, 1], [self.k0.value, self.k1.value])

        # Spatially varying diffusion coefficients for shorter patterns
        dA = np.interp(self.grid_mask, [0, 1], [self.dA0.value, self.dA1.value])
        dB = np.interp(self.grid_mask, [0, 1], [self.dB0.value, self.dB1.value])

        # Apply diffusion with spatially varying coefficients
        laplacian_a = cv2.filter2D(a, -1, kernel)
        laplacian_b = cv2.filter2D(b, -1, kernel)

        # Reaction-diffusion equations with spatially varying parameters
        self.grid_a = np.clip(a + self.delta * (dA * laplacian_a - ab2 + feed * (1 - a)), 0, 1)
        self.grid_b = np.clip(b + self.delta * (dB * laplacian_b + ab2 - (k + feed) * b), 0, 1)

    def draw(self):
        w, h = self.size
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if click[0] == 1 or click[2] == 1:
            x, y = mouse[0], mouse[1]

            # grid_a, grid_b interaction
            if self.g0.within_global(x, y):
                pos = self.g0.get_local(x, y)

                if click[0] == 1:
                    s = 5  # Smaller brush for finer control
                    cv2.circle(self.grid_a, pos, s, 0, -3)
                    cv2.circle(self.grid_b, pos, s, 1, -3)
                if click[2] == 1:
                    s = 5
                    cv2.circle(self.grid_a, pos, s, 1, -3)
                    cv2.circle(self.grid_b, pos, s, 0, -3)

        if click[0] == 1:
            for s in self.sliders:
                s.on_click(mouse)

        # Run multiple iterations for faster pattern evolution
        for i in range(10):
            self.calc_step()

        self.screen.fill((0, 0, 0))

        #### g0 - Main pattern display
        img = np.zeros((h, w, 3), dtype=np.int8)
        img[:, :, 0] = np.clip(self.grid_a * 255, 0, 255)
        img[:, :, 1] = 10
        img[:, :, 2] = np.clip(self.grid_b * 255, 0, 255)
        self.g0.blit(self.screen, img)

        #### g1 - Gradient mask visualization
        img = np.zeros((h, w, 3), dtype=np.int8)
        img[:, :, 0] = np.clip(self.grid_mask * 255, 0, 255)
        img[:, :, 1] = 0
        img[:, :, 2] = np.clip((1 - self.grid_mask) * 255, 0, 255)
        self.g1.blit(self.screen, img)
        self.g1.text(self.screen, "Press ESC to reset", (5, 5), self.font, pygame.Color("white"))

        #### g2 - Information display
        text = f"""FPS: {self.clock.get_fps():.1f}
        
Pit Pattern Types:
Type I → Type III → Type IV
(Normal → Small tubular → Gyrus-like)

A=Red (Activator), B=Blue (Inhibitor)
        
Presets:
1: Type I (normal) f={self.presets[0][0]:.3f}, k={self.presets[0][1]:.3f}
2: Type III (small) f={self.presets[1][0]:.3f}, k={self.presets[1][1]:.3f}
3: Type IV (gyrus) f={self.presets[2][0]:.3f}, k={self.presets[2][1]:.3f}"""

        self.g2.text(self.screen, text, (5, 5), self.font, pygame.Color("white"))

        #### g3 - Sliders
        for s in self.sliders:
            s.draw(self.screen, self.font)

        pygame.display.update()
        self.clock.tick()

    def start(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.reset_values()
                    if event.key == pygame.K_1:
                        self.set_preset(0)
                    if event.key == pygame.K_2:
                        self.set_preset(1)
                    if event.key == pygame.K_3:
                        self.set_preset(2)

            self.draw()
        pygame.quit()
        return


def main():
    dr = DiffusionReaction()
    dr.start()


if __name__ == "__main__":
    main()
