"""Vectorised fairy-dust physics (pure NumPy, no graphics dependency).

All particles live in fixed-size arrays ("struct of arrays") so thousands can be
simulated per frame with a handful of NumPy operations.

Particle kinds
    DUST          soft glowing mote that drifts and fades
    SPARKLE       4-point star that swells and fades quickly
    ANCHOR        static twinkling point (keeps a permanent drawing shimmering)
    ANCHOR_STAR   static point drawn as a strongly twinkling star
"""

import numpy as np

DUST, SPARKLE, ANCHOR, ANCHOR_STAR = 0, 1, 2, 3


class ParticleSystem:
    def __init__(self, capacity: int, seed=None):
        n = int(capacity)
        self.capacity = n
        self.rng = np.random.default_rng(seed)
        self.alive = np.zeros(n, dtype=bool)
        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.life = np.zeros(n, dtype=np.float32)
        self.life0 = np.ones(n, dtype=np.float32)
        self.size = np.zeros(n, dtype=np.float32)
        self.color = np.zeros(n, dtype=np.int16)
        self.kind = np.zeros(n, dtype=np.int8)
        self.grav = np.zeros(n, dtype=np.float32)
        self.drag = np.zeros(n, dtype=np.float32)
        self.turb = np.zeros(n, dtype=np.float32)
        self.phase = np.zeros(n, dtype=np.float32)
        self.tspeed = np.zeros(n, dtype=np.float32)
        self.age = np.zeros(n, dtype=np.float32)

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return int(self.alive.sum())

    def anchor_count(self) -> int:
        return int((self.alive & (self.kind >= ANCHOR)).sum())

    # ------------------------------------------------------------------
    def emit(self, pos, vel, life, size, color, kind, grav=0.0, drag=0.0, turb=0.0, tspeed=6.0):
        """Spawn particles. Scalars are broadcast; arrays must have one row per particle."""
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 2)
        n = len(pos)
        if n == 0:
            return

        def b(x, dtype, shape=None):
            return np.broadcast_to(np.asarray(x, dtype=dtype), shape or (n,))

        vel = b(vel, np.float32, (n, 2))
        life = b(life, np.float32)
        size = b(size, np.float32)
        color = b(color, np.int16)
        kind = b(kind, np.int8)
        grav = b(grav, np.float32)
        drag = b(drag, np.float32)
        turb = b(turb, np.float32)
        tspeed = b(tspeed, np.float32)

        free = np.flatnonzero(~self.alive)
        if len(free) >= n:
            slots = free[:n]
        else:
            need = n - len(free)
            cand = np.flatnonzero(self.alive & (self.kind < ANCHOR))   # never evict anchors
            if len(cand) > need:
                cand = cand[np.argsort(self.life[cand])[:need]]        # evict shortest-lived
            slots = np.concatenate([free, cand])
            n = len(slots)
            if n == 0:
                return

        sel = slice(0, n)
        self.pos[slots] = pos[sel]
        self.vel[slots] = vel[sel]
        self.life[slots] = life[sel]
        self.life0[slots] = np.where(np.isfinite(life[sel]), np.maximum(life[sel], 1e-3), 1.0)
        self.size[slots] = size[sel]
        self.color[slots] = color[sel]
        self.kind[slots] = kind[sel]
        self.grav[slots] = grav[sel]
        self.drag[slots] = drag[sel]
        self.turb[slots] = turb[sel]
        self.tspeed[slots] = tspeed[sel]
        self.phase[slots] = self.rng.uniform(0.0, 6.2832, n)
        self.age[slots] = 0.0
        self.alive[slots] = True

    def emit_anchors(self, pos, color, kind, size):
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 2)
        n = len(pos)
        if n:
            self.emit(pos, np.zeros((n, 2)), np.inf, size, color, kind,
                      tspeed=self.rng.uniform(2.0, 6.0, n))

    # ------------------------------------------------------------------
    def update(self, dt: float, t: float, bounds) -> None:
        a = self.alive
        if not a.any():
            return
        self.age[a] += dt

        idx = np.flatnonzero(a & (self.kind < ANCHOR))
        if idx.size == 0:
            return

        p = self.pos[idx]
        v = self.vel[idx]
        ph = self.phase[idx]
        tb = self.turb[idx]

        # Gentle swirling air currents plus gravity.
        ax = np.sin(p[:, 1] * 0.012 + t * 1.7 + ph) * tb
        ay = np.cos(p[:, 0] * 0.010 + t * 1.3 + ph * 1.7) * tb + self.grav[idx]
        v[:, 0] += ax * dt
        v[:, 1] += ay * dt
        v *= np.exp(-self.drag[idx] * dt)[:, None]
        p += v * dt

        self.pos[idx] = p
        self.vel[idx] = v
        life = self.life[idx] - dt
        self.life[idx] = life

        w, h = bounds
        off = (p[:, 0] < -60) | (p[:, 0] > w + 60) | (p[:, 1] < -60) | (p[:, 1] > h + 60)
        self.alive[idx[(life <= 0) | off]] = False

    # ------------------------------------------------------------------
    def pop_oldest_anchors(self, k: int):
        idx = np.flatnonzero(self.alive & (self.kind >= ANCHOR))
        if idx.size == 0 or k <= 0:
            return self._empty_pop()
        if idx.size > k:
            idx = idx[np.argsort(-self.age[idx])[:k]]
        return self._pop(idx)

    def pop_all_anchors(self):
        idx = np.flatnonzero(self.alive & (self.kind >= ANCHOR))
        return self._pop(idx) if idx.size else self._empty_pop()

    def _pop(self, idx):
        out = (self.pos[idx].copy(), self.color[idx].copy(), self.kind[idx].copy(), self.size[idx].copy())
        self.alive[idx] = False
        return out

    @staticmethod
    def _empty_pop():
        return (np.zeros((0, 2), np.float32), np.zeros(0, np.int16),
                np.zeros(0, np.int8), np.zeros(0, np.float32))

    # ------------------------------------------------------------------
    def render_data(self, t: float):
        """Per-particle drawing info, or None if nothing is alive.

        Returns a dict of arrays: x, y, color, size (glow radius in px),
        bright (0..1), star (bool: draw as 4-point star).
        """
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return None

        kind = self.kind[idx]
        life = self.life[idx]
        frac = np.where(np.isfinite(life), np.clip(life / self.life0[idx], 0.0, 1.0), 1.0)
        tw = 0.5 + 0.5 * np.sin(t * self.tspeed[idx] + self.phase[idx] * 3.0)   # 0..1 twinkle
        size = self.size[idx].astype(np.float64)

        bright = np.empty(idx.size, dtype=np.float64)

        m = kind == DUST
        bright[m] = frac[m] ** 0.7 * (0.55 + 0.45 * tw[m])

        m = kind == SPARKLE
        swell = np.sin(np.pi * (1.0 - frac[m]))
        bright[m] = np.clip(swell, 0, 1) ** 0.6 * (0.7 + 0.3 * tw[m])
        size[m] *= 0.5 + 0.7 * swell

        m = kind == ANCHOR
        bright[m] = 0.35 + 0.65 * tw[m]

        m = kind == ANCHOR_STAR
        bright[m] = 0.10 + 0.90 * tw[m] ** 2
        size[m] *= 0.7 + 0.5 * tw[m]

        pos = self.pos[idx]
        return {
            "x": pos[:, 0],
            "y": pos[:, 1],
            "color": self.color[idx].astype(np.int64),
            "size": size,
            "bright": np.clip(bright, 0.0, 1.0),
            "star": (kind == SPARKLE) | (kind == ANCHOR_STAR),
        }
