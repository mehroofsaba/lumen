# LUMEN — an interactive fairy dust experience

Draw in the air with your finger and watch your trail turn into shimmering,
drifting fairy dust. Python + OpenCV + MediaPipe + NumPy + Pygame.

## 1. Requirements

- A computer with a webcam (Windows, macOS or Linux)
- **Python 3.10, 3.11 or 3.12** (recommended; MediaPipe wheels may lag on newer versions)
- Internet access on first run only (to install packages and, with newer
  MediaPipe versions, download a ~8 MB hand model once)

## 2. Install and run

### Windows
Double-click `run_windows.bat`, or in a terminal:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python main.py

### macOS / Linux
    chmod +x run_unix.sh
    ./run_unix.sh

or manually:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python main.py

macOS: allow camera access for Terminal when prompted (System Settings →
Privacy & Security → Camera), then run again.

### No camera? Try it with the mouse
    python main.py --mouse

Left button draws, right button scatters dust.

### Options
    python main.py --camera 1        use a different camera
    python main.py --fullscreen      start fullscreen
    python main.py --mode fleeting   start in fleeting-trail mode
    python main.py --model PATH      use a local hand_landmarker.task file
    python main.py --no-help         hide the controls card at startup

## 3. Controls

| Gesture | Action |
|---|---|
| Index finger up | Draw |
| Pinch (thumb + index) | Pause / resume drawing |
| Open palm | Scatter fairy dust |
| Fist (hold ~0.6 s) | Clear the canvas |
| Two fingers (hold ~0.3 s) | Change particle effect |

Keyboard: `M` mode · `E` effect · `P` pause · `C` clear · `S` save picture ·
`F` fullscreen · `H` help · `Esc` quit. You can also click the
Permanent / Fleeting switch at the top right.

### The big decision: Permanent vs Fleeting
- **Permanent** — your drawing stays on the canvas, glowing, with twinkling
  stars that keep shimmering until you clear it.
- **Fleeting** — the dust drifts, swirls and fades away, leaving nothing behind.

Toggle any time with `M` or the on-screen switch.

## 4. Tuning

Everything is in `lumen_app/config.py`:
- Jitter vs. lag: `SMOOTH_MIN_CUTOFF` (lower = smoother), `SMOOTH_BETA` (higher = less lag)
- Hand travel needed to reach the screen edges: `ACTIVE_MARGIN_X/Y`
- Gesture sensitivity and hold times: `EXTEND_*`, `PINCH_*`, `*_DWELL`, `FIST_HOLD`
- Pinch behaviour: `PINCH_MODE = "toggle"` or `"hold"`
- Amount of dust and sparkle: `MAX_PARTICLES`, `ANCHOR_PROB`, `SCATTER_RATE`
- New effects: add an `Effect(...)` to the list in `lumen_app/effects.py`

If it runs slowly, lower `MAX_PARTICLES` and `MAX_ANCHORS`.

## 5. Tips for good tracking
- Face a window or lamp; avoid strong light behind you.
- Keep your hand ~40–80 cm from the camera, palm facing it.
- Draw with only the index finger extended; curl the others firmly.

## 6. Package as a desktop app

    pip install -r requirements-build.txt
    python build.py

The app appears in `dist/LUMEN/`. Build on each OS you want to target
(PyInstaller does not cross-compile). If your MediaPipe version needs the
separate model file, `build.py` bundles it when `models/hand_landmarker.task`
exists; otherwise the app downloads it on first launch.

## 7. Tests

    python -m tests.test_core

Covers gestures, smoothing, strokes, particles and both drawing modes without
needing a camera or window.

## 8. Troubleshooting
- **"Could not open camera"** — close other apps using it, check OS camera
  permission, try `--camera 1`.
- **"Could not download the hand model"** — download
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  and save it as `models/hand_landmarker.task`.
- **MediaPipe won't install** — use Python 3.10–3.12.
- **Gestures misfire** — adjust thresholds in `config.py`, improve lighting.
