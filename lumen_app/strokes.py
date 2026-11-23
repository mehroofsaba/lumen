"""Turns a stream of (smoothed) fingertip points into one continuous curve.

Tracking arrives at ~30 Hz, so successive points can be 30+ px apart at speed.
Joining them with straight lines looks polygonal. Instead each new point extends
the curve with a Catmull-Rom spline segment, sampled every `spacing` pixels.
The curve always reaches the newest point (no lag behind your fingertip).
"""

import math

import numpy as np

_EMPTY_PTS = np.zeros((0, 2))
_EMPTY_ARC = np.zeros((0,))


def catmull_rom(p0, p1, p2, p3, t):
    """Points on the spline between p1 and p2 for parameters t (array in 0..1)."""
    t = np.asarray(t, dtype=np.float64)[:, None]
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


class StrokeBuilder:
    def __init__(self, spacing: float = 4.0):
        self.spacing = float(spacing)
        self.reset()

    def reset(self) -> None:
        self._pts = []          # last few control points
        self.arc = 0.0          # arc length so far (used for colour cycling)

    @property
    def active(self) -> bool:
        return bool(self._pts)

    def add(self, point):
        """Add a control point. Returns (samples (M,2), arc_lengths (M,))."""
        p2 = np.asarray(point, dtype=np.float64)

        if not self._pts:                      # first point: leave a dot
            self._pts = [p2]
            return p2[None, :].copy(), np.array([0.0])

        p1 = self._pts[-1]
        seg = float(np.linalg.norm(p2 - p1))
        if seg < self.spacing * 0.25:          # ignore sub-pixel hand tremor
            return _EMPTY_PTS, _EMPTY_ARC

        p0 = self._pts[-2] if len(self._pts) >= 2 else p1
        p3 = 2.0 * p2 - p1                     # extrapolated "future" point
        n = max(1, int(math.ceil(seg / self.spacing)))
        t = (np.arange(n) + 1.0) / n           # excludes p1, includes p2
        samples = catmull_rom(p0, p1, p2, p3, t)
        arcs = self.arc + t * seg
        self.arc += seg

        self._pts.append(p2)
        if len(self._pts) > 3:
            self._pts.pop(0)
        return samples, arcs
