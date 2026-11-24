"""Draws the fairy dust with pygame.

Performance trick: every particle is a pre-rendered glow sprite (cached by
colour / size / brightness) blitted *additively*, so overlapping dust brightens
like real light. Thousands of sprites per frame are submitted in one batched call.
"""

import math

import numpy as np
import pygame

from . import config as C
from .effects import COLOR_TABLE

SIZE_RADII = np.array([2, 3, 4, 6, 8, 11, 15, 20])
N_SIZE = len(SIZE_RADII)
N_BRIGHT = 8
STAR_SCALE = 1.7
STAMP_SIZE_BUCKET = 3          # glow radius of the permanent line (6 px)


def _glow_pixels(radius: int, rgb, level: float, star: bool) -> np.ndarray:
    """RGB pixel array (rows, cols, 3) of one glow sprite."""
    R = max(2, int(round(radius * (STAR_SCALE if star else 1.0))))
    n = 2 * R + 1
    axis = np.arange(n) - R
    dx, dy = np.meshgrid(axis, axis)
    d = np.sqrt(dx * dx + dy * dy) / R

    if star:
        w = 0.07 * R + 0.6
        arm_h = np.exp(-(dy / w) ** 2) * np.clip(1 - np.abs(dx) / R, 0, 1) ** 1.6
        arm_v = np.exp(-(dx / w) ** 2) * np.clip(1 - np.abs(dy) / R, 0, 1) ** 1.6
        core = np.exp(-(d / 0.22) ** 2)
        g = np.clip(arm_h + arm_v + 0.9 * core + 0.25 * np.exp(-(d / 0.5) ** 2), 0, 1)
        white = core
    else:
        edge = np.clip(1.0 - d, 0, 1)
        g = np.exp(-(d / 0.5) ** 2) * edge ** 0.5
        white = np.exp(-(d / 0.2) ** 2)

    base = np.asarray(rgb, dtype=np.float64) / 255.0
    out = np.clip(base[None, None, :] * g[..., None] + 0.55 * white[..., None], 0, 1) * level
    return (out * 255).astype(np.uint8)


class Renderer:
    def __init__(self, size):
        self.w, self.h = int(size[0]), int(size[1])
        self._cache = {}
        self._fblits = hasattr(pygame.Surface, "fblits")

        self.bg = self._make_background()
        self.layer = self._blank()
        self.fading = None
        self._fade_left = 1.2
        self._seen_clear = 0

        rng = np.random.default_rng(7)
        self._stars = [
            (int(rng.integers(0, self.w)), int(rng.integers(0, self.h)),
             float(rng.uniform(0, 6.28)), float(rng.uniform(0.6, 2.2)), int(rng.choice([1, 1, 1, 2])))
            for _ in range(90)
        ]

    # ------------------------------------------------------------------
    def _blank(self):
        s = pygame.Surface((self.w, self.h)).convert()
        s.fill((0, 0, 0))
        return s

    def _make_background(self):
        h, w = self.h, self.w
        y = np.linspace(0.0, 1.0, h)[:, None]
        x = np.linspace(-1.0, 1.0, w)[None, :]
        top = np.array([9.0, 7.0, 26.0])
        bottom = np.array([27.0, 11.0, 46.0])
        grad = top * (1.0 - y[..., None]) + bottom * y[..., None]            # (h,1,3)
        vignette = 1.0 - 0.38 * ((x ** 2 + (y * 2 - 1) ** 2) / 2.0)          # (h,w)
        arr = np.clip(grad * vignette[..., None], 0, 255).astype(np.uint8)    # (h,w,3)
        return pygame.image.frombuffer(arr.tobytes(), (w, h), "RGB").convert()

    def _sprite(self, color_id: int, size_bucket: int, bright_bucket: int, star: bool):
        key = (color_id, size_bucket, bright_bucket, star)
        entry = self._cache.get(key)
        if entry is None:
            if len(self._cache) > 6000:
                self._cache.clear()
            level = (bright_bucket + 1) / N_BRIGHT
            px = _glow_pixels(int(SIZE_RADII[size_bucket]), COLOR_TABLE[color_id], level, star)
            n = px.shape[0]
            surf = pygame.image.frombuffer(px.tobytes(), (n, n), "RGB").convert()
            entry = (surf, n // 2)
            self._cache[key] = entry
        return entry

    def _blit_add(self, target, items) -> None:
        if not items:
            return
        flag = pygame.BLEND_RGB_ADD
        if self._fblits:
            target.fblits(items, flag)
        else:
            target.blits([(s, p, None, flag) for s, p in items], False)

    # ------------------------------------------------------------------
    def _consume(self, engine, dt: float) -> None:
        if engine.clear_count != self._seen_clear:
            self._seen_clear = engine.clear_count
            self.fading = self.layer            # old drawing melts away...
            self._fade_left = 1.2
            self.layer = self._blank()          # ...while a fresh canvas starts

        for pts, colors in engine.stamps:
            items = []
            for (x, y), c in zip(pts.tolist(), colors.tolist()):
                surf, half = self._sprite(c, STAMP_SIZE_BUCKET, C.STAMP_BRIGHTNESS_LEVEL, False)
                items.append((surf, (int(x) - half, int(y) - half)))
            self._blit_add(self.layer, items)
        engine.stamps.clear()

        for pos, col, kind, size in engine.bakes:
            items = []
            for (x, y), c, k, s in zip(pos.tolist(), col.tolist(), kind.tolist(), size.tolist()):
                star = k == 3
                sb = min(int(np.searchsorted(SIZE_RADII, s)), N_SIZE - 1)
                surf, half = self._sprite(c, sb, 5 if not star else 6, star)
                items.append((surf, (int(x) - half, int(y) - half)))
            self._blit_add(self.layer, items)
        engine.bakes.clear()

    def _draw_stars(self, screen, t: float) -> None:
        for x, y, ph, sp, r in self._stars:
            b = 0.30 + 0.70 * (0.5 + 0.5 * math.sin(t * sp + ph))
            c = int(140 * b)
            pygame.draw.circle(screen, (c, c, min(255, c + 45)), (x, y), r)

    def _draw_particles(self, screen, engine, t: float) -> None:
        data = engine.particles.render_data(t)
        if data is None:
            return
        vis = data["bright"] > 0.05
        if not vis.any():
            return
        # Keep particle attributes in NumPy arrays until the blit list is built.
        # Converting every masked array to Python lists creates several large
        # temporary objects on every frame, especially in dual-hand mode.
        indices = np.flatnonzero(vis)
        sizes = np.minimum(np.searchsorted(SIZE_RADII, data["size"]), N_SIZE - 1)
        brightness = np.clip(
            (data["bright"] * N_BRIGHT).astype(np.int64), 0, N_BRIGHT - 1
        )

        items = []
        append = items.append
        sprite = self._sprite
        xs, ys = data["x"], data["y"]
        colors, stars = data["color"], data["star"]
        for i in indices:
            surf, half = sprite(
                int(colors[i]), int(sizes[i]), int(brightness[i]), bool(stars[i])
            )
            append((surf, (int(xs[i]) - half, int(ys[i]) - half)))
        self._blit_add(screen, items)

    def _draw_cursor(self, screen, engine, t: float) -> None:
        if not engine.hand_visible:
            return
        fx = engine.effect
        ids = fx.color_ids
        cid = ids[int(t * 1.5) % len(ids)] if fx.cycle else ids[0]
        color = COLOR_TABLE[cid]
        x, y = int(engine.cursor[0]), int(engine.cursor[1])
        pulse = 0.5 + 0.5 * math.sin(t * 7.0)

        if engine.drawing:
            items = []
            for key in ((cid, 5, 7, False), (cid, 4, 7, True)):
                surf, half = self._sprite(*key)
                items.append((surf, (x - half, y - half)))
            self._blit_add(screen, items)
        elif engine.is_paused:
            pygame.draw.circle(screen, (130, 130, 170), (x, y), 13, 2)
            pygame.draw.line(screen, (130, 130, 170), (x - 4, y - 5), (x - 4, y + 5), 2)
            pygame.draw.line(screen, (130, 130, 170), (x + 4, y - 5), (x + 4, y + 5), 2)
        else:
            pygame.draw.circle(screen, color, (x, y), int(10 + 2 * pulse), 2)

        if engine.gesture.name == "PALM":
            px, py = int(engine.palm[0]), int(engine.palm[1])
            pygame.draw.circle(screen, color, (px, py), int(38 + 6 * pulse), 2)
        if engine.fist_progress > 0.0:
            rect = pygame.Rect(x - 28, y - 28, 56, 56)
            end = -math.pi / 2 + engine.fist_progress * 2 * math.pi
            pygame.draw.arc(screen, (255, 235, 200), rect, -math.pi / 2, end, 4)

    # ------------------------------------------------------------------
    def draw(self, screen, engine, dt: float, t: float) -> None:
        screen.blit(self.bg, (0, 0))
        self._draw_stars(screen, t)
        self._consume(engine, dt)

        screen.blit(self.layer, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if self.fading is not None:
            self.fading.fill((234, 234, 234), special_flags=pygame.BLEND_RGB_MULT)
            screen.blit(self.fading, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            self._fade_left -= dt
            if self._fade_left <= 0:
                self.fading = None

        self._draw_particles(screen, engine, t)
        self._draw_cursor(screen, engine, t)
