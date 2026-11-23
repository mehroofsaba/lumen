"""LumenEngine: all of LUMEN's behaviour, with no graphics or camera dependency.

The engine receives hand landmarks (or pointer input), decides what the user is
doing, and maintains the particle world. The renderer then simply draws whatever
the engine holds. Keeping this layer pure NumPy makes it easy to test headlessly.
"""

import math
from enum import Enum
from typing import List, Optional

import numpy as np

from . import config as C
from .effects import EFFECTS, pick_colors
from .gestures import Gesture, GestureEngine, GestureState
from .particles import ANCHOR, ANCHOR_STAR, DUST, SPARKLE, ParticleSystem
from .smoothing import OneEuroFilter
from .strokes import StrokeBuilder

TWO_PI = 2.0 * math.pi


class Mode(Enum):
    PERMANENT = "permanent"   # drawings stay and keep shimmering until cleared
    FLEETING = "fleeting"     # trails fade away like real fairy dust


class LumenEngine:
    def __init__(self, size=C.WINDOW_SIZE, seed: Optional[int] = None):
        self.w, self.h = int(size[0]), int(size[1])
        self.rng = np.random.default_rng(seed)

        self.gestures = GestureEngine()
        self.tip_filter = OneEuroFilter(C.SMOOTH_MIN_CUTOFF, C.SMOOTH_BETA, C.SMOOTH_D_CUTOFF)
        self.palm_filter = OneEuroFilter(C.SMOOTH_MIN_CUTOFF, C.SMOOTH_BETA, C.SMOOTH_D_CUTOFF)
        self.stroke = StrokeBuilder(C.TRAIL_SPACING)
        self.particles = ParticleSystem(C.MAX_PARTICLES + C.MAX_ANCHORS + 400, seed=seed)

        self.mode = Mode(C.DEFAULT_MODE)
        self.effect_index = 0
        self.paused = False

        # Live state (read by the renderer / HUD)
        self.cursor = np.array([self.w / 2.0, self.h / 2.0])
        self.cursor_vel = np.zeros(2)
        self.palm = np.array([self.w / 2.0, self.h / 2.0])
        self.hand_visible = False
        self.gesture = Gesture.NONE
        self.fist_progress = 0.0
        self.drawing = False
        self.has_drawn = False

        # Hand-offs to the renderer (drained every frame)
        self.stamps: list = []        # (points (M,2), colour ids (M,)) -> glowing line
        self.bakes: list = []         # popped anchors -> baked into the canvas
        self.clear_count = 0

        self.toasts: List[list] = []  # [text, seconds_left]
        self.t = 0.0                  # simulation clock
        self._last_obs_t: Optional[float] = None
        self._last_obs_time = -1e9

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def effect(self):
        return EFFECTS[self.effect_index]

    @property
    def is_paused(self) -> bool:
        held = C.PINCH_MODE == "hold" and self.gesture == Gesture.PINCH
        return self.paused or held

    # ------------------------------------------------------------------
    # Actions (used by gestures, keyboard and on-screen buttons)
    # ------------------------------------------------------------------
    def toast(self, text: str, seconds: float = 2.2) -> None:
        self.toasts = [x for x in self.toasts if x[0] != text][-2:]
        self.toasts.append([text, seconds])

    def set_mode(self, mode: Mode) -> None:
        if mode != self.mode:
            self.mode = mode
            self._end_stroke()
            self.toast("Permanent drawings" if mode == Mode.PERMANENT else "Fleeting fairy trails")

    def toggle_mode(self) -> None:
        self.set_mode(Mode.FLEETING if self.mode == Mode.PERMANENT else Mode.PERMANENT)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._end_stroke()
        self.toast("Drawing paused" if self.paused else "Drawing resumed")

    def cycle_effect(self, step: int = 1) -> None:
        self.effect_index = (self.effect_index + step) % len(EFFECTS)
        self.toast(f"Effect: {self.effect.name}")
        if self.hand_visible:
            self._burst(self.cursor, 55, power=170.0)

    def clear(self) -> None:
        """Dissolve the whole canvas into drifting dust."""
        pos, col, kind, size = self.particles.pop_all_anchors()
        n = len(pos)
        if n:
            ang = self.rng.uniform(0, TWO_PI, n)
            speed = self.rng.uniform(50, 260, n)
            vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * speed[:, None]
            is_star = kind == ANCHOR_STAR
            self.particles.emit(
                pos, vel, self.rng.uniform(0.8, 1.9, n),
                np.where(is_star, size * 1.4, size * 1.3), col,
                np.where(is_star, SPARKLE, DUST),
                grav=-8.0, drag=1.4, turb=40.0, tspeed=self.rng.uniform(3, 9, n),
            )
        self.stamps.clear()
        self.bakes.clear()
        self.clear_count += 1
        self._end_stroke()
        self.toast("Canvas cleared")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def _to_canvas(self, xy) -> np.ndarray:
        mx, my = C.ACTIVE_MARGIN_X, C.ACTIVE_MARGIN_Y
        x = (float(xy[0]) - mx) / (1.0 - 2.0 * mx)
        y = (float(xy[1]) - my) / (1.0 - 2.0 * my)
        return np.array([min(max(x, 0.0), 1.0) * self.w, min(max(y, 0.0), 1.0) * self.h])

    def feed_hand(self, landmarks: Optional[np.ndarray], t: float, aspect: Optional[float] = None) -> None:
        """Camera observation. `landmarks` is (21, 3) normalised, or None if no hand."""
        if landmarks is None:
            return    # the hand-lost grace timer in update() takes care of this
        if aspect:
            self.gestures.aspect = float(aspect)

        tip = self._to_canvas(landmarks[8][:2])
        palm = self._to_canvas(np.asarray(landmarks)[[0, 5, 9, 13, 17], :2].mean(axis=0))

        if not self.hand_visible:                 # hand (re)appeared: no smoothing glide
            self.tip_filter.reset()
            self.palm_filter.reset()
            self.gestures.reset()
            self._last_obs_t = None
        self.hand_visible = True
        self._last_obs_time = t

        tip_s = self.tip_filter(tip, t)
        palm_s = self.palm_filter(palm, t)
        state = self.gestures.update(landmarks, t)
        self._apply(state, tip_s, palm_s, t)

    def feed_pointer(self, pos, gesture: Gesture, t: float) -> None:
        """Mouse / fallback input: `pos` in canvas pixels, or None if idle."""
        if pos is None:
            return
        if not self.hand_visible:
            self._last_obs_t = None
        self.hand_visible = True
        self._last_obs_time = t
        p = np.asarray(pos, dtype=np.float64)
        self._apply(GestureState(gesture=gesture, raw=gesture), p, p, t)

    def _apply(self, state: GestureState, tip: np.ndarray, palm: np.ndarray, t: float) -> None:
        self.gesture = state.gesture
        self.fist_progress = state.fist_progress

        for ev in state.events:
            if ev == "toggle_pause":
                self.toggle_pause()
            elif ev == "cycle_effect":
                self.cycle_effect()
            elif ev == "clear":
                self.clear()

        # Cursor velocity (for letting dust inherit a little of your motion)
        if self._last_obs_t is not None and t > self._last_obs_t:
            inst = (tip - self.cursor) / (t - self._last_obs_t)
            self.cursor_vel = 0.6 * self.cursor_vel + 0.4 * inst
        self._last_obs_t = t
        self.cursor = tip
        self.palm = palm

        if state.gesture == Gesture.DRAW and not self.is_paused:
            self._draw_to(tip)
        else:
            self._end_stroke()

    def _lose_hand(self) -> None:
        self.hand_visible = False
        self.gesture = Gesture.NONE
        self.fist_progress = 0.0
        self.cursor_vel[:] = 0.0
        self._end_stroke()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_to(self, p: np.ndarray) -> None:
        if self.stroke.active and self.drawing:
            last = self.stroke._pts[-1]
            if float(np.linalg.norm(p - last)) > C.MAX_STROKE_JUMP * math.hypot(self.w, self.h):
                self.stroke.reset()           # tracking glitch: don't draw a long line
        self.drawing = True
        self.has_drawn = True
        pts, arcs = self.stroke.add(p)
        if len(pts):
            self._emit_trail(pts, arcs)

    def _end_stroke(self) -> None:
        self.drawing = False
        self.stroke.reset()

    def _emit_trail(self, pts: np.ndarray, arcs: np.ndarray) -> None:
        fx = self.effect
        rng = self.rng
        m = len(pts)
        colors = pick_colors(fx, arcs, self.t, rng)

        counts = rng.poisson(fx.density, m)
        total = int(counts.sum())
        if total:
            base = np.repeat(pts, counts, axis=0)
            cols = np.repeat(colors, counts)
            pos = base + rng.normal(0, 2.2, (total, 2))
            ang = rng.uniform(0, TWO_PI, total)
            spd = np.abs(rng.normal(fx.speed, fx.speed * 0.5, total))
            vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * spd[:, None] + self.cursor_vel * 0.06
            self._spawn(pos, vel, cols)

        if self.mode == Mode.PERMANENT:
            self.stamps.append((pts.copy(), colors.copy()))
            keep = rng.random(m) < C.ANCHOR_PROB
            k = int(keep.sum())
            if k:
                star = rng.random(k) < C.ANCHOR_STAR_FRACTION
                self.particles.emit_anchors(
                    pts[keep] + rng.normal(0, 1.6, (k, 2)),
                    colors[keep],
                    np.where(star, ANCHOR_STAR, ANCHOR),
                    np.where(star, rng.uniform(3.5, 6.5, k), rng.uniform(1.6, 3.4, k)),
                )

    def _spawn(self, pos, vel, cols, life_scale=1.0, size_scale=1.0, sparkle_boost=0.0) -> None:
        fx = self.effect
        rng = self.rng
        n = len(pos)
        life = rng.uniform(fx.life[0], fx.life[1], n) * life_scale
        size = rng.uniform(fx.size[0], fx.size[1], n) * size_scale
        spark = rng.random(n) < (fx.sparkle + sparkle_boost)
        life = np.where(spark, rng.uniform(0.5, 1.2, n), life)
        size = np.where(spark, rng.uniform(4.0, 10.0, n), size)
        self.particles.emit(
            pos, vel, life, size, cols, spark.astype(np.int8),
            grav=fx.gravity, drag=fx.drag, turb=fx.turbulence,
            tspeed=rng.uniform(3.0, 9.0, n),
        )

    def _burst(self, center, n: int, power: float = 260.0) -> None:
        if n <= 0:
            return
        rng = self.rng
        ang = rng.uniform(0, TWO_PI, n)
        spd = power * (0.25 + rng.gamma(2.0, 0.4, n))
        vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * spd[:, None]
        pos = np.asarray(center, dtype=np.float64) + rng.normal(0, 12, (n, 2))
        cols = pick_colors(self.effect, rng.uniform(0, 2000, n), self.t, rng)
        self._spawn(pos, vel, cols, life_scale=1.1, size_scale=1.15, sparkle_boost=0.12)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(self, dt: float, now: float) -> None:
        dt = min(max(dt, 0.0), 0.05)
        self.t += dt

        if self.hand_visible and now - self._last_obs_time > C.HAND_LOST_GRACE:
            self._lose_hand()

        if self.hand_visible:
            self.cursor_vel *= math.exp(-dt * 2.5)

            if self.gesture == Gesture.PALM:
                n = int(self.rng.poisson(C.SCATTER_RATE * dt))
                self._burst(self.palm, n, power=230.0)
            elif (self.drawing and not self.is_paused
                  and float(np.linalg.norm(self.cursor_vel)) < 40.0):
                n = int(self.rng.poisson(C.IDLE_DUST_RATE * dt))
                if n:
                    pos = self.cursor + self.rng.normal(0, 3.0, (n, 2))
                    ang = self.rng.uniform(0, TWO_PI, n)
                    vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * 18.0
                    self._spawn(pos, vel, pick_colors(self.effect, self.rng.uniform(0, 2000, n), self.t, self.rng))

        self.particles.update(dt, self.t, (self.w, self.h))

        over = self.particles.anchor_count() - C.MAX_ANCHORS
        if over > 0:
            self.bakes.append(self.particles.pop_oldest_anchors(over + C.ANCHOR_BAKE_BATCH))

        for item in self.toasts:
            item[1] -= dt
        self.toasts = [x for x in self.toasts if x[1] > 0]
