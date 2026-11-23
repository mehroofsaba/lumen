"""Particle effects ("two fingers" cycles through these).

All colours live in one global table so each particle only needs to store a small
integer colour id; the renderer turns ids into cached glow sprites.
"""

import colorsys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

COLOR_TABLE: List[Tuple[int, int, int]] = []


def _register(rgb) -> int:
    COLOR_TABLE.append(tuple(int(c) for c in rgb))
    return len(COLOR_TABLE) - 1


def _palette(colors):
    return tuple(_register(c) for c in colors)


def _rainbow(n=24, sat=0.55):
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, sat, 1.0)
        out.append((r * 255, g * 255, b * 255))
    return out


@dataclass(frozen=True)
class Effect:
    name: str
    blurb: str
    color_ids: Tuple[int, ...]
    cycle: bool            # walk smoothly through the palette along the stroke
    cycle_arc: float       # palette cycles per pixel of stroke
    cycle_time: float      # palette cycles per second
    density: float         # dust particles per trail sample (every ~4 px)
    speed: float           # initial drift speed (px/s)
    life: Tuple[float, float]
    size: Tuple[float, float]
    gravity: float         # px/s^2, positive falls, negative rises
    drag: float
    turbulence: float      # strength of the swirling air currents
    sparkle: float         # chance a particle is a 4-point sparkle


EFFECTS: List[Effect] = [
    Effect(
        name="Stardust", blurb="warm golden glitter",
        color_ids=_palette([(255, 214, 120), (255, 236, 170), (255, 190, 110),
                            (255, 165, 195), (255, 248, 225), (250, 205, 255)]),
        cycle=False, cycle_arc=0, cycle_time=0,
        density=1.6, speed=22, life=(0.9, 2.2), size=(1.4, 3.8),
        gravity=22, drag=1.6, turbulence=30, sparkle=0.10,
    ),
    Effect(
        name="Moonlight", blurb="cool silver frost",
        color_ids=_palette([(200, 225, 255), (235, 245, 255), (170, 200, 255),
                            (215, 190, 255), (255, 255, 255), (150, 230, 255)]),
        cycle=False, cycle_arc=0, cycle_time=0,
        density=1.3, speed=16, life=(1.2, 2.8), size=(1.2, 3.4),
        gravity=10, drag=1.2, turbulence=22, sparkle=0.14,
    ),
    Effect(
        name="Aurora", blurb="shifting northern lights",
        color_ids=_palette([(60, 255, 200), (80, 255, 150), (80, 210, 255),
                            (120, 150, 255), (170, 110, 255), (230, 100, 255),
                            (255, 110, 200), (230, 100, 255), (170, 110, 255),
                            (120, 150, 255), (80, 210, 255), (80, 255, 150)]),
        cycle=True, cycle_arc=0.0016, cycle_time=0.08,
        density=1.8, speed=26, life=(1.0, 2.6), size=(1.6, 4.2),
        gravity=-14, drag=1.1, turbulence=60, sparkle=0.08,
    ),
    Effect(
        name="Fireflies", blurb="slow rising embers",
        color_ids=_palette([(255, 240, 120), (220, 255, 110), (255, 200, 80),
                            (190, 255, 140), (255, 225, 150)]),
        cycle=False, cycle_arc=0, cycle_time=0,
        density=0.9, speed=14, life=(1.8, 3.8), size=(2.2, 5.0),
        gravity=-20, drag=0.9, turbulence=45, sparkle=0.05,
    ),
    Effect(
        name="Rainbow", blurb="a full spectrum ribbon",
        color_ids=_palette(_rainbow()),
        cycle=True, cycle_arc=0.0022, cycle_time=0.12,
        density=1.7, speed=24, life=(0.9, 2.2), size=(1.4, 3.8),
        gravity=28, drag=1.5, turbulence=30, sparkle=0.12,
    ),
]


def pick_colors(fx: Effect, arcs: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    """Colour ids for trail samples at the given arc lengths."""
    ids = np.asarray(fx.color_ids, dtype=np.int16)
    n = len(arcs)
    if fx.cycle:
        phase = (np.asarray(arcs) * fx.cycle_arc + t * fx.cycle_time) % 1.0
        k = (phase * len(ids)).astype(np.int64)
        jitter = rng.integers(-1, 2, size=n) * (rng.random(n) < 0.25)
        return ids[(k + jitter) % len(ids)]
    return ids[rng.integers(0, len(ids), size=n)]
