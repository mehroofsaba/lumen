"""Turns 21 hand landmarks into stable, debounced gestures.

Gestures (the proposed set from the project brief):
    DRAW   index finger only          -> draw
    PINCH  thumb touches index tip    -> pause / resume drawing
    PALM   open palm                  -> scatter fairy dust
    FIST   closed fist (hold)         -> clear canvas
    TWO    index + middle finger      -> change particle effect

Robustness comes from four things: rotation-independent finger tests, hysteresis
(a finger must clearly bend to stop counting as extended), a short confirmation
window, and dwell times so passing through a pose never triggers an action.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np

from . import config as C

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# (pip joint, tip) for index, middle, ring, pinky
_FINGERS = ((INDEX_PIP, INDEX_TIP), (MIDDLE_PIP, MIDDLE_TIP),
            (RING_PIP, RING_TIP), (PINKY_PIP, PINKY_TIP))


class Gesture(IntEnum):
    NONE = 0
    DRAW = 1
    PINCH = 2
    PALM = 3
    FIST = 4
    TWO = 5


GESTURE_LABELS = {
    Gesture.NONE: "",
    Gesture.DRAW: "Drawing",
    Gesture.PINCH: "Pinch",
    Gesture.PALM: "Scatter",
    Gesture.FIST: "Clear",
    Gesture.TWO: "Effect",
}


@dataclass
class GestureState:
    gesture: Gesture = Gesture.NONE      # stable (debounced) gesture
    raw: Gesture = Gesture.NONE          # this frame's raw classification
    events: List[str] = field(default_factory=list)   # one-shot actions
    fist_progress: float = 0.0           # 0..1 while holding a fist


class GestureEngine:
    def __init__(self, aspect: float = C.CAMERA_SIZE[0] / C.CAMERA_SIZE[1]):
        # Landmarks are normalised per-axis; multiplying x by the frame aspect
        # ratio makes distances behave like real (isotropic) distances.
        self.aspect = float(aspect)
        self.reset()

    def reset(self) -> None:
        self._ext = [False, False, False, False]
        self._pinching = False
        self._cand = Gesture.NONE
        self._cand_n = 0
        self.stable = Gesture.NONE
        self._stable_since = 0.0
        self._fired = False
        self._last_toggle = -1e9
        self._last_effect = -1e9

    # ------------------------------------------------------------------
    def classify(self, lm: np.ndarray) -> Gesture:
        p = np.asarray(lm, dtype=np.float64)[:, :2] * np.array([self.aspect, 1.0])
        wrist = p[WRIST]

        def dist(a, b):
            return float(np.linalg.norm(p[a] - p[b]))

        hand_size = max(dist(WRIST, MIDDLE_MCP), 1e-6)

        ratios = []
        for i, (pip, tip) in enumerate(_FINGERS):
            r = float(np.linalg.norm(p[tip] - wrist)) / max(float(np.linalg.norm(p[pip] - wrist)), 1e-6)
            ratios.append(r)
            threshold = C.EXTEND_EXIT if self._ext[i] else C.EXTEND_ENTER
            self._ext[i] = r > threshold

        pinch_ratio = dist(THUMB_TIP, INDEX_TIP) / hand_size
        threshold = C.PINCH_EXIT if self._pinching else C.PINCH_ENTER
        self._pinching = pinch_ratio < threshold and ratios[0] > C.PINCH_INDEX_MIN
        if self._pinching:
            return Gesture.PINCH

        index, middle, ring, pinky = self._ext
        if index and not (middle or ring or pinky):
            return Gesture.DRAW
        if index and middle and not (ring or pinky):
            return Gesture.TWO
        if index and middle and ring and pinky:
            return Gesture.PALM
        if not (index or middle or ring or pinky):
            return Gesture.FIST
        return Gesture.NONE

    # ------------------------------------------------------------------
    def update(self, lm: Optional[np.ndarray], now: float) -> GestureState:
        raw = self.classify(lm) if lm is not None else Gesture.NONE

        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = raw, 1

        if self._cand != self.stable and self._cand_n >= C.CONFIRM_FRAMES:
            self.stable = self._cand
            self._stable_since = now
            self._fired = False

        events: List[str] = []
        fist_progress = 0.0
        dwell = now - self._stable_since
        g = self.stable

        if g == Gesture.PINCH:
            if (C.PINCH_MODE == "toggle" and not self._fired and dwell >= C.PINCH_DWELL
                    and now - self._last_toggle >= C.TOGGLE_COOLDOWN):
                events.append("toggle_pause")
                self._fired = True
                self._last_toggle = now
        elif g == Gesture.TWO:
            if (not self._fired and dwell >= C.TWO_FINGER_DWELL
                    and now - self._last_effect >= C.EFFECT_COOLDOWN):
                events.append("cycle_effect")
                self._fired = True
                self._last_effect = now
        elif g == Gesture.FIST:
            fist_progress = min(1.0, dwell / C.FIST_HOLD)
            if fist_progress >= 1.0 and not self._fired:
                events.append("clear")
                self._fired = True

        return GestureState(gesture=g, raw=raw, events=events, fist_progress=fist_progress)
