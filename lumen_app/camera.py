"""Camera capture + MediaPipe hand tracking on a background thread.

Keeping capture and detection off the main thread means the visuals stay at a
smooth 60 fps even when hand tracking runs at the camera's ~30 fps.

Two MediaPipe back-ends are supported automatically:
  * the classic `mediapipe.solutions.hands` (models bundled with the package)
  * the newer Tasks API `HandLandmarker` (needs a small model file, which is
    downloaded once on first run and cached)
"""

import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from . import config as C

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


# ----------------------------------------------------------------------------
# Paths / model file
# ----------------------------------------------------------------------------
def project_root() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def user_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".lumen")


def _model_dirs():
    return [os.path.join(project_root(), "models"), user_dir()]


def find_model(override: Optional[str] = None) -> Optional[str]:
    if override:
        return override if os.path.isfile(override) else None
    for d in _model_dirs():
        path = os.path.join(d, C.MODEL_NAME)
        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            return path
    return None


def download_model(progress: Optional[Callable[[str], None]] = None) -> str:
    """Download the hand model once. Returns the local path."""
    last_error = None
    for d in _model_dirs():
        try:
            os.makedirs(d, exist_ok=True)
            target = os.path.join(d, C.MODEL_NAME)
            tmp = target + ".part"
            with urllib.request.urlopen(C.MODEL_URL, timeout=30) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(f"Downloading hand model… {100 * done // total}%")
            if os.path.getsize(tmp) < 1_000_000:
                raise RuntimeError("downloaded file is too small")
            os.replace(tmp, target)
            return target
        except Exception as exc:           # try the next folder
            last_error = exc
    raise RuntimeError(
        f"Could not download the hand model ({last_error}). Download it manually from\n"
        f"{C.MODEL_URL}\nand save it as models/{C.MODEL_NAME} next to main.py."
    )


# ----------------------------------------------------------------------------
# Detector back-ends
# ----------------------------------------------------------------------------
class _LegacyDetector:
    def __init__(self, mp):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=C.DETECT_CONFIDENCE,
            min_tracking_confidence=C.TRACKING_CONFIDENCE,
        )

    def detect_all(self, rgb: np.ndarray, ts_ms: int) -> list:
        res = self._hands.process(rgb)

        if not res.multi_hand_landmarks:
            return []

        hands = []

        for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
            landmarks = np.array(
                [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
                dtype=np.float32,
            )

            label = "Unknown"
            if res.multi_handedness and i < len(res.multi_handedness):
                classification = res.multi_handedness[i].classification
                if classification:
                    label = classification[0].label

            hands.append({
                "label": label,
                "landmarks": landmarks,
            })

        return hands

    def detect(self, rgb: np.ndarray, ts_ms: int) -> Optional[np.ndarray]:
        hands = self.detect_all(rgb, ts_ms)
        return hands[0]["landmarks"] if hands else None

    def close(self):
        self._hands.close()


class _TasksDetector:
    def __init__(self, mp, model_path: str):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp

        with open(model_path, "rb") as fh:
            buffer = fh.read()

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=buffer),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=C.DETECT_CONFIDENCE,
            min_hand_presence_confidence=C.PRESENCE_CONFIDENCE,
            min_tracking_confidence=C.TRACKING_CONFIDENCE,
        )

        self._lm = mp_vision.HandLandmarker.create_from_options(options)

    def detect_all(self, rgb: np.ndarray, ts_ms: int) -> list:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        res = self._lm.detect_for_video(image, ts_ms)

        if not res.hand_landmarks:
            return []

        hands = []

        for i, hand_landmarks in enumerate(res.hand_landmarks):
            landmarks = np.array(
                [[p.x, p.y, p.z] for p in hand_landmarks],
                dtype=np.float32,
            )

            label = "Unknown"
            if res.handedness and i < len(res.handedness):
                categories = res.handedness[i]
                if categories:
                    label = categories[0].category_name

            hands.append({
                "label": label,
                "landmarks": landmarks,
            })

        return hands

    def detect(self, rgb: np.ndarray, ts_ms: int) -> Optional[np.ndarray]:
        hands = self.detect_all(rgb, ts_ms)
        return hands[0]["landmarks"] if hands else None

    def close(self):
        self._lm.close()


def create_detector(progress: Callable[[str], None], model_override: Optional[str] = None):
    import mediapipe as mp          # heavy import: done on the worker thread

    legacy_error = None
    if not model_override and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        try:
            return _LegacyDetector(mp)
        except Exception as exc:    # fall through to the Tasks API
            legacy_error = exc

    path = find_model(model_override)
    if path is None:
        if model_override:
            raise RuntimeError(f"Model file not found: {model_override}")
        path = download_model(progress)
    try:
        return _TasksDetector(mp, path)
    except Exception as exc:
        extra = f" (classic API also failed: {legacy_error})" if legacy_error else ""
        raise RuntimeError(f"MediaPipe could not start: {exc}{extra}")


# ----------------------------------------------------------------------------
# Worker thread
# ----------------------------------------------------------------------------
@dataclass
class Observation:
    seq: int = 0
    t: float = 0.0
    landmarks: Optional[np.ndarray] = None
    preview: Optional[np.ndarray] = None
    aspect: float = 4 / 3
    hands: Optional[list] = None


class CameraWorker(threading.Thread):
    """state: 'starting' -> 'loading' -> 'running'   (or 'error')"""

    def __init__(self, index: int = C.CAMERA_INDEX, model_override: Optional[str] = None):
        super().__init__(daemon=True, name="lumen-camera")
        self.index = index
        self.model_override = model_override
        self.cap = None
        self.state = "starting"
        self.message = "Starting camera…"
        self.fps = 0.0
        self._lock = threading.Lock()
        self._obs = Observation()
        self._stop_event = threading.Event()

    # -- main-thread API ---------------------------------------------------
    def open_capture(self) -> bool:
        """Open the camera. Call from the main thread (macOS permission prompt needs it)."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAMERA_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_SIZE[1])
                cap.set(cv2.CAP_PROP_FPS, C.CAMERA_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                return True
            cap.release()
        self._fail(f"Could not open camera {self.index}. Close other apps using it, "
                   f"check permissions, or try --camera 1.")
        return False

    def latest(self) -> Observation:
        with self._lock:
            return self._obs

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def failed(self) -> bool:
        return self.state == "error"

    # -- internals ---------------------------------------------------------
    def _set(self, state: str, message: str) -> None:
        self.state, self.message = state, message

    def _fail(self, message: str) -> None:
        self._set("error", message)

    def _make_preview(self, rgb: np.ndarray, hands: list) -> np.ndarray:
        pw, ph = C.PREVIEW_SIZE
        small = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)

        colors = {
            "Left": (255, 180, 120),
            "Right": (180, 220, 255),
            "Unknown": (255, 214, 140),
        }

        for hand in hands:
            lm = hand["landmarks"]
            label = hand["label"]
            color = colors.get(label, colors["Unknown"])

            pts = [(int(x * pw), int(y * ph)) for x, y, _ in lm]

            for a, b in HAND_CONNECTIONS:
                cv2.line(small, pts[a], pts[b], color, 2, cv2.LINE_AA)

            for pt in pts:
                cv2.circle(small, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

            cv2.circle(
                small, pts[8], 6, (255, 120, 220), 2, cv2.LINE_AA
            )

            cv2.putText(
                small,
                label,
                (pts[0][0], max(15, pts[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return small

    def run(self) -> None:
        detector = None
        try:
            self._set("loading", "Loading hand tracker…")
            detector = create_detector(lambda msg: self._set("loading", msg), self.model_override)
            if self.cap is None:
                raise RuntimeError("camera is not open")
            self._set("running", "Show your hand")

            t0 = time.perf_counter()
            last_ts = -1
            seq = 0
            frames = 0
            fps_t = t0
            bad_reads = 0

            while not self._stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    bad_reads += 1
                    if bad_reads > 90:
                        raise RuntimeError("camera stopped delivering frames")
                    time.sleep(0.01)
                    continue
                bad_reads = 0

                frame = cv2.flip(frame, 1)                       # mirror: feels like a mirror
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = int((time.perf_counter() - t0) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                hands = detector.detect_all(rgb, ts)

                # Keep the first hand available for the existing single-hand engine.
                lm = hands[0]["landmarks"] if hands else None

                preview = self._make_preview(rgb, hands)

                seq += 1
                obs = Observation(
                    seq=seq,
                    t=time.perf_counter(),
                    landmarks=lm,
                    preview=preview,
                    aspect=rgb.shape[1] / rgb.shape[0],
                    hands=hands,
                )
                with self._lock:
                    self._obs = obs

                frames += 1
                now = time.perf_counter()
                if now - fps_t >= 1.0:
                    self.fps = frames / (now - fps_t)
                    frames, fps_t = 0, now
        except Exception as exc:
            self._fail(str(exc))
        finally:
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            if self.cap is not None:
                self.cap.release()
