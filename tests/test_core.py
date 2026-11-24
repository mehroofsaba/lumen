"""Headless tests for LUMEN's logic (no camera, window or pygame needed).

Run with:   python -m tests.test_core        (or:  pytest -q)
"""

import math

import numpy as np

from lumen_app import config as C
from lumen_app.effects import COLOR_TABLE, EFFECTS, pick_colors
from lumen_app.engine import HandMode, LumenEngine, Mode
from lumen_app.gestures import Gesture, GestureEngine
from lumen_app.particles import ANCHOR, DUST, ParticleSystem
from lumen_app.smoothing import OneEuroFilter
from lumen_app.strokes import StrokeBuilder

DT = 1 / 30


# ---------------------------------------------------------------------------
# Synthetic hands (normalised landmarks, 21 x 3)
# ---------------------------------------------------------------------------
def make_hand(ext=(1, 0, 0, 0), pinch=False, tip=(0.44, 0.32)):
    """Index/middle/ring/pinky extended flags; index tip is placed at `tip`."""
    lm = np.zeros((21, 2))
    lm[0] = (0.5, 0.8)
    mcps = [(0.44, 0.55), (0.50, 0.54), (0.56, 0.55), (0.62, 0.58)]
    for i, (mx, my) in enumerate(mcps):
        base = 5 + 4 * i
        lm[base] = (mx, my)
        if ext[i]:
            lm[base + 1] = (mx, my - 0.09)
            lm[base + 2] = (mx, my - 0.16)
            lm[base + 3] = (mx, my - 0.23)
        else:
            lm[base + 1] = (mx, my - 0.07)
            lm[base + 2] = (mx, my - 0.02)
            lm[base + 3] = (mx, my + 0.02)
    lm[1], lm[2], lm[3] = (0.43, 0.74), (0.38, 0.68), (0.34, 0.63)
    lm[4] = (0.30, 0.60)                                   # thumb tip, away from index
    if pinch:
        lm[4] = lm[8] + (0.01, 0.02)
    elif not any(ext):
        lm[4] = lm[8] + (0.0, 0.01)                        # fist: thumb tucked near index
    lm += np.array(tip) - lm[8]
    return np.hstack([lm, np.zeros((21, 1))])


HANDS = {
    Gesture.DRAW: lambda **k: make_hand((1, 0, 0, 0), **k),
    Gesture.TWO: lambda **k: make_hand((1, 1, 0, 0), **k),
    Gesture.PALM: lambda **k: make_hand((1, 1, 1, 1), **k),
    Gesture.FIST: lambda **k: make_hand((0, 0, 0, 0), **k),
    Gesture.PINCH: lambda **k: make_hand((1, 0, 0, 0), pinch=True, **k),
}


def feed(engine, gesture, seconds, t0, tip=(0.44, 0.32)):
    """Feed `seconds` of a held gesture; returns the end time."""
    n = max(1, int(round(seconds / DT)))
    for i in range(n):
        t = t0 + i * DT
        engine.feed_hand(HANDS[gesture](tip=tip), t, aspect=1.0)
        engine.update(DT, t)
    return t0 + n * DT


# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------
def test_classifier_recognises_every_pose():
    g = GestureEngine(aspect=1.0)
    for gesture, make in HANDS.items():
        g.reset()
        assert g.classify(make()) == gesture, f"expected {gesture.name}"


def test_fist_with_tucked_thumb_is_not_a_pinch():
    g = GestureEngine(aspect=1.0)
    assert g.classify(HANDS[Gesture.FIST]()) == Gesture.FIST


def test_debounce_ignores_single_frame_glitch():
    g = GestureEngine(aspect=1.0)
    t = 0.0
    for _ in range(5):
        st = g.update(HANDS[Gesture.DRAW](), t); t += DT
    assert st.gesture == Gesture.DRAW
    st = g.update(HANDS[Gesture.PALM](), t); t += DT      # one stray frame
    assert st.gesture == Gesture.DRAW
    st = g.update(HANDS[Gesture.DRAW](), t)
    assert st.gesture == Gesture.DRAW


# ---------------------------------------------------------------------------
# Smoothing and strokes
# ---------------------------------------------------------------------------
def test_one_euro_filter_removes_jitter_but_follows_motion():
    rng = np.random.default_rng(1)
    f = OneEuroFilter(C.SMOOTH_MIN_CUTOFF, C.SMOOTH_BETA, C.SMOOTH_D_CUTOFF)
    out = []
    for i in range(300):                                   # hand held still, noisy tracking
        out.append(f(np.array([500.0, 300.0]) + rng.normal(0, 3.0, 2), i * DT))
    out = np.array(out[60:])
    assert out.std(axis=0).max() < 1.6, "stationary jitter should be strongly reduced"

    f.reset()
    last = None
    for i in range(90):                                    # fast sweep: should keep up
        last = f(np.array([100.0 + i * 20.0, 300.0]), i * DT)
    assert abs(last[0] - (100.0 + 89 * 20.0)) < 80, "filter must not lag far behind fast motion"


def test_stroke_samples_are_evenly_spaced_and_reach_the_newest_point():
    s = StrokeBuilder(spacing=4.0)
    last_pts = None
    all_pts = []
    for i in range(40):
        a = i * 0.25
        p = np.array([400 + 200 * math.cos(a), 300 + 200 * math.sin(a)])
        pts, arcs = s.add(p)
        if len(pts):
            all_pts.append(pts)
            last_pts = pts
            assert np.all(np.diff(arcs) >= 0)
        assert np.allclose(last_pts[-1], p)
    pts = np.vstack(all_pts)
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert gaps.max() < 8.5, "no visible gaps in the trail"


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------
def test_particles_fall_fade_and_die():
    ps = ParticleSystem(100, seed=0)
    ps.emit(np.array([[100.0, 100.0]] * 10), np.zeros((10, 2)), 0.5, 3.0, 0, DUST, grav=50.0, drag=0.5)
    y0 = ps.pos[ps.alive][:, 1].mean()
    for i in range(10):
        ps.update(0.02, i * 0.02, (1280, 720))
    assert ps.pos[ps.alive][:, 1].mean() > y0, "gravity pulls dust down"
    for i in range(40):
        ps.update(0.02, 1 + i * 0.02, (1280, 720))
    assert ps.count == 0, "dust expires"


def test_overflowing_the_pool_never_crashes_and_keeps_anchors():
    ps = ParticleSystem(50, seed=0)
    ps.emit_anchors(np.random.rand(10, 2) * 100, 0, ANCHOR, 2.0)
    ps.emit(np.random.rand(200, 2) * 100, np.zeros((200, 2)), 1.0, 2.0, 0, DUST)
    assert ps.anchor_count() == 10
    assert ps.count == 50
    assert ps.render_data(0.0) is not None


def test_render_data_is_well_formed():
    ps = ParticleSystem(64, seed=0)
    ps.emit(np.random.rand(20, 2) * 500, np.zeros((20, 2)), 1.0, 3.0, 2, 1)       # sparkles
    ps.emit_anchors(np.random.rand(10, 2) * 500, 1, 3, 4.0)                        # star anchors
    d = ps.render_data(1.234)
    n = len(d["x"])
    assert n == 30 and all(len(d[k]) == n for k in d)
    assert d["bright"].min() >= 0 and d["bright"].max() <= 1


def test_every_color_id_is_valid():
    for fx in EFFECTS:
        for cid in fx.color_ids:
            assert 0 <= cid < len(COLOR_TABLE)
        ids = pick_colors(fx, np.linspace(0, 3000, 50), 2.0, np.random.default_rng(0))
        assert set(ids.tolist()) <= set(fx.color_ids)


# ---------------------------------------------------------------------------
# Engine behaviour end-to-end
# ---------------------------------------------------------------------------
def draw_circle(engine, seconds=2.0, t0=0.0):
    n = int(seconds / DT)
    for i in range(n):
        a = i * 0.12
        tip = (0.5 + 0.25 * math.cos(a), 0.5 + 0.22 * math.sin(a))
        t = t0 + i * DT
        engine.feed_hand(HANDS[Gesture.DRAW](tip=tip), t, aspect=1.0)
        engine.update(DT, t)
    return t0 + n * DT


def test_permanent_mode_leaves_lasting_twinkling_drawings():
    e = LumenEngine(seed=3)
    e.set_mode(Mode.PERMANENT)
    draw_circle(e)
    assert e.drawing and e.has_drawn
    assert e.stamps, "glowing line is handed to the renderer"
    assert e.particles.anchor_count() > 50
    # anchors never expire
    n = e.particles.anchor_count()
    for i in range(300):
        e.update(0.016, 100 + i * 0.016)       # hand vanishes, time passes
    assert e.particles.anchor_count() == n


def test_fleeting_mode_leaves_nothing_behind():
    e = LumenEngine(seed=3)
    e.set_mode(Mode.FLEETING)
    t = draw_circle(e)
    assert e.particles.count > 50
    assert e.particles.anchor_count() == 0 and not e.stamps
    for i in range(400):                        # ~6.5 s later all dust has faded
        e.update(0.016, t + 1 + i * 0.016)
    assert e.particles.count == 0


def test_toggle_mode_roundtrip():
    e = LumenEngine(seed=0)
    start = e.mode
    e.toggle_mode(); e.toggle_mode()
    assert e.mode == start

def test_hand_mode_switch_preserves_existing_artwork():
    e = LumenEngine(seed=7)
    e.set_mode(Mode.PERMANENT)

    draw_circle(e, 1.5)
    anchors_before = e.particles.anchor_count()
    assert anchors_before > 0

    e.set_hand_mode(HandMode.DUAL)
    assert e.hand_mode == HandMode.DUAL
    assert e.particles.anchor_count() == anchors_before

    e.set_hand_mode(HandMode.SINGLE)
    assert e.hand_mode == HandMode.SINGLE
    assert e.particles.anchor_count() == anchors_before

def test_dual_hand_tracking_keeps_hand_states_independent():
    e = LumenEngine(seed=8)
    e.set_hand_mode(HandMode.DUAL)

    left = make_hand(ext=(1, 0, 0, 0), tip=(0.3, 0.4))
    right = make_hand(ext=(1, 0, 0, 0), tip=(0.7, 0.4))

    hands = [
        {"label": "Left", "landmarks": left},
        {"label": "Right", "landmarks": right},
    ]

    for i in range(20):
        t = i * DT
        e.feed_hands(hands, t, aspect=1.0)
        e.update(DT, t)

    assert e.hands["Left"].visible
    assert e.hands["Right"].visible
    assert e.hands["Left"].drawing
    assert e.hands["Right"].drawing
    assert e.drawing    

def test_dual_hand_loss_does_not_interrupt_other_hand():
    e = LumenEngine(seed=9)
    e.set_hand_mode(HandMode.DUAL)

    left = make_hand(tip=(0.3, 0.4))
    right = make_hand(tip=(0.7, 0.4))

    both_hands = [
        {"label": "Left", "landmarks": left},
        {"label": "Right", "landmarks": right},
    ]

    for i in range(20):
        t = i * DT
        e.feed_hands(both_hands, t, aspect=1.0)
        e.update(DT, t)

    assert e.hands["Left"].drawing
    assert e.hands["Right"].drawing

    # Only the right hand remains visible.
    # Keep sending fresh observations for it while the left hand is missing.
    for i in range(int(C.HAND_LOST_GRACE / DT) + 4):
        t += DT
        e.feed_hands(
            [{"label": "Right", "landmarks": right}],
            t,
            aspect=1.0,
        )
        e.update(DT, t)

    assert not e.hands["Left"].visible
    assert not e.hands["Left"].drawing
    assert e.hands["Right"].visible
    assert e.hands["Right"].drawing
    assert e.drawing    

def test_single_hand_mode_keeps_selected_hand_locked():
    e = LumenEngine(seed=10)
    e.set_hand_mode(HandMode.SINGLE)

    left = make_hand(tip=(0.3, 0.4))
    right = make_hand(tip=(0.7, 0.4))

    # Left is detected first, so it should be selected.
    e.feed_hands(
        [
            {"label": "Left", "landmarks": left},
            {"label": "Right", "landmarks": right},
        ],
        0.0,
        aspect=1.0,
    )

    assert e._single_hand_label == "Left"

    # Detection order changes, but selection should remain Left.
    e.feed_hands(
        [
            {"label": "Right", "landmarks": right},
            {"label": "Left", "landmarks": left},
        ],
        DT,
        aspect=1.0,
    )

    assert e._single_hand_label == "Left"
    assert e.hands["Left"].visible
    assert not e.hands["Right"].visible

def test_single_hand_mode_reacquires_other_hand_after_timeout():
    e = LumenEngine(seed=11)
    e.set_hand_mode(HandMode.SINGLE)

    left = make_hand(tip=(0.3, 0.4))
    right = make_hand(tip=(0.7, 0.4))

    e.feed_hands(
        [{"label": "Left", "landmarks": left}],
        0.0,
        aspect=1.0,
    )
    assert e._single_hand_label == "Left"

    # Let the selected hand disappear beyond the grace period.
    e.update(0.016, C.HAND_LOST_GRACE + 0.2)

    assert not e.hands["Left"].visible
    assert e._single_hand_label is None

    # The right hand appears and should now be selected.
    t = C.HAND_LOST_GRACE + 0.2 + DT
    e.feed_hands(
        [{"label": "Right", "landmarks": right}],
        t,
        aspect=1.0,
    )

    assert e._single_hand_label == "Right"
    assert e.hands["Right"].visible    

def test_pinch_toggles_pause_once_per_pinch():
    e = LumenEngine(seed=1)
    t = feed(e, Gesture.DRAW, 0.5, 0.0)
    assert not e.paused
    t = feed(e, Gesture.PINCH, 0.6, t)
    assert e.paused, "pinch pauses"
    t = feed(e, Gesture.DRAW, 0.8, t)
    assert e.paused and not e.drawing, "stays paused until pinched again"
    t = feed(e, Gesture.PINCH, 0.6, t)
    assert not e.paused, "second pinch resumes"
    feed(e, Gesture.DRAW, 0.5, t)
    assert e.drawing


def test_two_fingers_cycles_effect_once_and_ignores_passing_poses():
    e = LumenEngine(seed=1)
    t = feed(e, Gesture.DRAW, 0.4, 0.0)
    assert e.effect_index == 0
    t = feed(e, Gesture.TWO, 0.2, t)            # brief pass-through: below dwell time
    t = feed(e, Gesture.PALM, 0.4, t)
    assert e.effect_index == 0, "transition through two fingers must not trigger"
    t = feed(e, Gesture.TWO, 0.6, t)
    assert e.effect_index == 1
    t = feed(e, Gesture.TWO, 0.6, t)
    assert e.effect_index == 1, "holding does not keep cycling"
    t = feed(e, Gesture.DRAW, 0.4, t)
    feed(e, Gesture.TWO, 0.6, t + 1.5)
    assert e.effect_index == 2


def test_fist_must_be_held_to_clear():
    e = LumenEngine(seed=2)
    e.set_mode(Mode.PERMANENT)
    t = draw_circle(e, 1.5)
    assert e.particles.anchor_count() > 0
    t = feed(e, Gesture.FIST, 0.35, t)
    assert e.clear_count == 0 and 0 < e.fist_progress < 1, "not yet"
    t = feed(e, Gesture.FIST, 0.6, t)
    assert e.clear_count == 1 and e.particles.anchor_count() == 0
    assert any("cleared" in x[0] for x in e.toasts)


def test_open_palm_scatters_dust_without_drawing():
    e = LumenEngine(seed=4)
    feed(e, Gesture.PALM, 1.0, 0.0, tip=(0.5, 0.4))
    assert e.particles.count > 40
    assert not e.stamps and e.particles.anchor_count() == 0


def test_hand_loss_ends_the_stroke_after_grace_period():
    e = LumenEngine(seed=5)
    t = draw_circle(e, 0.6)
    assert e.drawing
    e.update(0.016, t + 0.05)
    assert e.hand_visible, "short dropouts are tolerated"
    e.update(0.016, t + C.HAND_LOST_GRACE + 0.2)
    assert not e.hand_visible and not e.drawing


def test_big_tracking_jump_does_not_draw_a_long_line():
    e = LumenEngine(seed=6)
    t = feed(e, Gesture.DRAW, 0.5, 0.0, tip=(0.2, 0.3))
    before = len(e.stamps)
    e.stamps.clear()
    t = feed(e, Gesture.DRAW, 0.1, t, tip=(0.8, 0.7))     # teleport
    longest = max((len(p) for p, _ in e.stamps), default=0)
    assert longest < 60, "no 300+ px streak across the screen"


def test_anchor_overflow_bakes_oldest_into_the_canvas():
    e = LumenEngine(seed=7)
    n = C.MAX_ANCHORS + 200
    e.particles.emit_anchors(np.random.rand(n, 2) * 1000, 0, ANCHOR, 2.0)
    e.update(0.016, 0.0)
    assert e.bakes and e.particles.anchor_count() <= C.MAX_ANCHORS


def test_all_effects_can_draw():
    for i in range(len(EFFECTS)):
        e = LumenEngine(seed=i)
        e.effect_index = i
        draw_circle(e, 0.8)
        assert e.particles.count > 0


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
