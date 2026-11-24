# LUMEN

**An interactive fairy dust experience.**

LUMEN lets you draw in the air using hand gestures. Your movements become glowing trails, drifting particles and sparkling effects.

Built with **Python, OpenCV, MediaPipe, NumPy and Pygame**.

## Features

- Draw in the air using hand tracking.
- Choose **Single Hand** or **Dual Hand** mode.
- Switch between **Permanent** and **Fleeting** trails.
- Explore five particle effects: Stardust, Moonlight, Aurora, Fireflies and Rainbow.
- Save your artwork as an image.
- Use mouse controls when a camera is unavailable.

## Get started

### Requirements

- Windows, macOS or Linux
- Python **3.10–3.12**
- A webcam for hand tracking

### Windows

Double-click `run_windows.bat`.

Or run these commands from the project folder:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

### macOS / Linux

```bash
chmod +x run_unix.sh
./run_unix.sh
```

The first launch needs an internet connection to install dependencies and, if required by your MediaPipe version, download the hand-tracking model.

## How to use LUMEN

Show your hand to the camera and use these gestures:

| Gesture | Action |
|---|---|
| Index finger up | Draw |
| Pinch thumb and index finger | Pause or resume |
| Open palm | Scatter particles |
| Hold a fist | Clear the canvas |
| Two fingers | Change particle effect |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `1` / `2` | Single / Dual Hand |
| `M` | Switch Permanent / Fleeting |
| `E` | Change particle effect |
| `P` | Pause or resume |
| `C` | Clear the canvas |
| `S` | Save your artwork |
| `F` | Toggle fullscreen |
| `H` | Show or hide the quick guide |
| `Esc` | Quit |

You can also use the on-screen controls to change drawing style and hand-tracking mode.

## No webcam?

Start LUMEN in mouse mode:

```bash
python main.py --mouse
```

- Hold the **left mouse button** to draw.
- Hold the **right mouse button** to scatter particles.

## Troubleshooting

- **Camera not found:** Close other apps using the webcam, check camera permissions, or try `python main.py --camera 1`.
- **Hand not detected:** Improve the lighting, face the camera and keep your hand clearly in view.
- **MediaPipe installation fails:** Use Python 3.10, 3.11 or 3.12.

---

Made with curiosity, code and a little fairy dust. ✨
