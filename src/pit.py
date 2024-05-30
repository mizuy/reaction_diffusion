# -*- coding: utf-8 -*-
import sys

import cv2
import numpy as np
import pygame
from pygame.locals import *

# https://algorithm.joho.info/programming/python/opencv-spatial-filtering-py/

# feed = 0.055; k = 0.062;
# feed =  0.0545; k = 0.062; //coral growth
# dA = 0.08; dB = 0.04; feed = 0.09; k = 0.06;
# dA = 0.004; dB = 0.0009; feed =  0.09; k = 0.056;
# dA = 0.004; dB = 0.0009; feed =  0.1; k = 0.06;


kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])


# https://stackoverflow.com/questions/42014195/rendering-text-with-multiple-lines-in-pygame
def blit_text(surface, text, pos, font, color=pygame.Color("black")):
    words = [word.split(" ") for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(" ")[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 1, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.


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

        # scale
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


# https://stackoverflow.com/questions/42014195/rendering-text-with-multiple-lines-in-pygame
def blit_text(surface, text, pos, font, color=pygame.Color("black")):
    words = [word.split(" ") for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(" ")[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 1, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.


# Pythonゲームプログラミング　知っておきたい数学と物理の基本
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
        # clip
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
    def __init__(self, dA=1, dB=0.5, width=256, height=256):
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
        pygame.display.set_caption("Diffusion Reaction")
        self.font = pygame.font.Font(None, 30)
        self.clock = pygame.time.Clock()

        self.dA = dA
        self.dB = dB
        self.delta = 1

        # feed, k
        self.presets = [(0.055, 0.062), (0.045, 0.066), (0.033, 0.056)]
        feed, k = self.presets[0]

        variables = [
            ("dA", "dA:{0:.2f}", 0.1, 2, 1, "white"),
            ("dB", "dB:{0:.2f}", 0.1, 2, 0.5, "white"),
            ("feed0", "feed0:{0:.3f}", 0.02, 0.07, feed, "Blue"),
            ("k0", "k0:{0:.3f}", 0.020, 0.07, k, "Blue"),
            ("feed1", "feed1:{0:.3f}", 0.020, 0.07, feed, "Red"),
            ("k1", "k1:{0:.3f}", 0.02, 0.07, k, "Red"),
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

        self.grid_mask = np.zeros((h, w))
        self.grid_a = np.ones((h, w))
        self.grid_b = np.zeros((h, w))
        # self.grid_b[100:110, 100:110] = 1

    def reset_values(self):
        w, h = self.w, self.h
        self.grid_mask = np.zeros((h, w))

    def set_preset(self, preset_i):
        feed, k = self.presets[preset_i]
        self.feed0.value = feed
        self.k0.value = k

    def calc_step(self):
        a = self.grid_a
        b = self.grid_b
        ab2 = a * (b**2)

        feed = self.feed0.value + self.grid_mask * (self.feed1.value - self.feed0.value)
        k = self.k0.value + self.grid_mask * (self.k1.value - self.k0.value)

        self.grid_a = np.clip(
            a + self.delta * (self.dA.value * cv2.filter2D(a, -1, kernel) - ab2 + feed * (1 - a)), 0, 1
        )
        self.grid_b = np.clip(
            b + self.delta * (self.dB.value * cv2.filter2D(b, -1, kernel) + ab2 - (k + feed) * b), 0, 1
        )

    def draw(self):
        w, h = self.size
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if click[0] == 1 or click[2] == 1:
            x, y = mouse[0], mouse[1]

            # grid_a, grid_b
            if self.g0.within_global(x, y):
                pos = self.g0.get_local(x, y)

                if click[0] == 1:
                    s = 10
                    cv2.circle(self.grid_a, pos, s, 0, -5)
                    cv2.circle(self.grid_b, pos, s, 1, -5)
                if click[2] == 1:
                    s = 10
                    cv2.circle(self.grid_a, pos, s, 1, -5)
                    cv2.circle(self.grid_b, pos, s, 0, -5)

            # grid_mask
            if self.g1.within_global(x, y):
                pos = self.g1.get_local(x, y)
                s = 10
                if click[0] == 1:
                    cv2.circle(self.grid_mask, pos, s, 1, -5)
                if click[2] == 1:
                    cv2.circle(self.grid_mask, pos, s, 0, -5)

        if click[0] == 1:
            for s in self.sliders:
                s.on_click(mouse)

        for i in range(10):
            self.calc_step()

        self.screen.fill((0, 0, 0))

        #### g0
        # render grid_a, grid_b
        img = np.zeros((h, w, 3), dtype=np.int8)
        img[:, :, 0] = np.clip(self.grid_a * 255, 0, 255)
        img[:, :, 1] = 10
        img[:, :, 2] = np.clip(self.grid_b * 255, 0, 255)
        self.g0.blit(self.screen, img)

        #### g1
        img = np.zeros((h, w, 3), dtype=np.int8)
        img[:, :, 0] = np.clip(self.grid_mask * 255, 0, 255)
        img[:, :, 1] = 0
        img[:, :, 2] = np.clip((1 - self.grid_mask) * 255, 0, 255)
        self.g1.blit(self.screen, img)
        self.g1.text(self.screen, "Press ESC to reset", (5, 5), self.font, pygame.Color("white"))

        #### g2
        text = f"""FPS: {self.clock.get_fps():.1f}
        
A=Red, B=Blue
        
Presets:
1: feed={self.presets[0][0]:.3f}, k={self.presets[0][1]:.3f}
2: feed={self.presets[1][0]:.3f}, k={self.presets[1][1]:.3f}
3: feed={self.presets[2][0]:.3f}, k={self.presets[2][1]:.3f}"""

        self.g2.text(self.screen, text, (5, 5), self.font, pygame.Color("white"))

        #### g3 sliders
        for s in self.sliders:
            s.draw(self.screen, self.font)

        pygame.display.update()
        self.clock.tick()

    def start(self):
        running = True
        while running:
            for event in pygame.event.get():  # 終了処理
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

                """'
                keys = pygame.key.get_pressed()
                if keys[pygame.K_i]:
                    self.feed0 -= 0.001
                """

            self.draw()
        pygame.quit()
        return


def main():
    dr = DiffusionReaction()
    dr.start()


if __name__ == "__main__":
    main()
