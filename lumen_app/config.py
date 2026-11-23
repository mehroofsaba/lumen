"""Tunable settings for LUMEN.

Everything you might want to tweak while "tuning" the experience lives here.
Gestures, smoothing and particle behaviour are all controlled from this file.
"""

APP_NAME = "LUMEN"
APP_TAGLINE = "an interactive fairy dust experience"

# --------------------------------------------------------------------------
# Window / timing
# --------------------------------------------------------------------------
WINDOW_SIZE = (1280, 720)      # logical size; the window can be resized/fullscreen
TARGET_FPS = 60

# --------------------------------------------------------------------------
# Camera and hand tracking
# --------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_SIZE = (640, 480)
CAMERA_FPS = 30
PREVIEW_SIZE = (256, 192)      # small camera panel shown in the corner

MODEL_NAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

DETECT_CONFIDENCE = 0.6
PRESENCE_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# Fraction of the camera frame (each edge) that is "dead space". Your hand only
# needs to travel across the middle of the frame to reach every screen edge.
ACTIVE_MARGIN_X = 0.12
ACTIVE_MARGIN_Y = 0.16

# --------------------------------------------------------------------------
# Smoothing (One Euro filter, applied in screen pixels)
#   min_cutoff  lower  = smoother when moving slowly (less jitter)
#   beta        higher = less lag when moving fast
# --------------------------------------------------------------------------
SMOOTH_MIN_CUTOFF = 2.8
SMOOTH_BETA = 0.02
SMOOTH_D_CUTOFF = 1.0

# Longest hop (as a fraction of the screen diagonal) that still counts as the
# same stroke. Bigger jumps start a new stroke instead of drawing a long line.
MAX_STROKE_JUMP = 0.30

# If the hand vanishes for this long the current stroke is ended.
HAND_LOST_GRACE = 0.25

# --------------------------------------------------------------------------
# Gesture recognition
# --------------------------------------------------------------------------
# A finger counts as "extended" when (wrist->tip) / (wrist->middle joint)
# exceeds the ENTER ratio, and stays extended until it drops below EXIT.
EXTEND_ENTER = 1.25
EXTEND_EXIT = 1.08

# Pinch = thumb tip close to index tip, measured relative to hand size.
PINCH_ENTER = 0.22
PINCH_EXIT = 0.32
PINCH_INDEX_MIN = 0.95         # index must not be curled into a fist

CONFIRM_FRAMES = 3             # frames a gesture must persist to become "stable"

# Dwell times (seconds a gesture must be held before its action fires). These
# prevent accidental triggers while your hand moves between poses.
PINCH_DWELL = 0.12
TWO_FINGER_DWELL = 0.30
FIST_HOLD = 0.60               # hold a fist this long to clear the canvas

TOGGLE_COOLDOWN = 0.70
EFFECT_COOLDOWN = 0.90

# "toggle": pinch once to pause, pinch again to resume.
# "hold"  : drawing is paused only while you keep pinching.
PINCH_MODE = "toggle"

# --------------------------------------------------------------------------
# Drawing and particles
# --------------------------------------------------------------------------
DEFAULT_MODE = "permanent"     # "permanent" or "fleeting"

TRAIL_SPACING = 4.0            # pixels between trail samples
MAX_PARTICLES = 4000           # short-lived dust and sparkles
MAX_ANCHORS = 1600             # twinkling points that keep permanent lines alive
ANCHOR_BAKE_BATCH = 150        # oldest anchors are baked into the canvas in batches
ANCHOR_PROB = 0.30             # chance a trail sample leaves a twinkling anchor
ANCHOR_STAR_FRACTION = 0.14    # share of anchors drawn as 4-point stars
STAMP_BRIGHTNESS_LEVEL = 2     # 0-7 brightness of the permanent glowing line

SCATTER_RATE = 170             # particles/second while the palm is open
IDLE_DUST_RATE = 30            # particles/second while drawing but not moving
