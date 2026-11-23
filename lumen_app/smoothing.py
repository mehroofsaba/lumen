"""Jitter-free tracking: a One Euro filter implemented with NumPy.

The One Euro filter (Casiez et al., 2012) adapts its cutoff to hand speed:
heavy smoothing when you hold still (kills jitter) and light smoothing when you
move quickly (keeps latency low). That is exactly what a drawing hand needs.
"""

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Smooths a vector signal (e.g. an (x, y) position) over time."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._x_hat = None
        self._dx_hat = None
        self._x_raw = None
        self._t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_hat is None:
            self._x_hat = x.copy()
            self._dx_hat = np.zeros_like(x)
            self._x_raw = x.copy()
            self._t = t
            return self._x_hat.copy()

        dt = t - self._t
        if dt <= 1e-6:
            return self._x_hat.copy()
        self._t = t

        # Smoothed speed estimate.
        dx = (x - self._x_raw) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat

        # Cutoff grows with speed -> less smoothing (less lag) when moving fast.
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx_hat))
        a = _alpha(cutoff, dt)
        self._x_hat = a * x + (1.0 - a) * self._x_hat
        self._x_raw = x.copy()
        return self._x_hat.copy()
