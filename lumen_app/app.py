"""LUMEN main application: window, input and main loop."""

import argparse
import os
import time
from types import SimpleNamespace

import pygame

from . import config as C
from .engine import LumenEngine, Mode
from .gestures import Gesture
from .render import Renderer
from .ui import Hud


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="lumen", description="LUMEN - an interactive fairy dust experience")
    p.add_argument("--camera", type=int, default=C.CAMERA_INDEX, help="camera index (default 0)")
    p.add_argument("--mouse", action="store_true", help="skip the camera; draw with the mouse")
    p.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    p.add_argument("--model", default=None, help="path to hand_landmarker.task (Tasks API only)")
    p.add_argument("--mode", choices=["permanent", "fleeting"], default=C.DEFAULT_MODE)
    p.add_argument("--no-help", action="store_true", help="hide the controls panel at startup")
    return p.parse_args(argv)


class LumenApp:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("LUMEN - Interactive Fairy Dust")
        flags = pygame.SCALED | pygame.RESIZABLE
        if args.fullscreen:
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode(C.WINDOW_SIZE, flags, vsync=1)
        except (TypeError, pygame.error):
            self.screen = pygame.display.set_mode(C.WINDOW_SIZE, flags)

        self.clock = pygame.time.Clock()
        self.engine = LumenEngine(C.WINDOW_SIZE)
        self.engine.mode = Mode(args.mode)
        self.renderer = Renderer(C.WINDOW_SIZE)
        self.hud = Hud(C.WINDOW_SIZE)
        self.hud.show_help = not args.no_help

        self.mouse_mode = bool(args.mouse)
        self.camera = None
        self._seq = 0
        self._cam_view = SimpleNamespace(preview=None, hand=False, state="starting", message="", fps=0.0)

        if not self.mouse_mode:
            self._start_camera(args)

        self.running = True
        self._help_dismissed = False

    # ------------------------------------------------------------------
    def _start_camera(self, args) -> None:
        try:
            from .camera import CameraWorker
            worker = CameraWorker(args.camera, args.model)
            if worker.open_capture():
                worker.start()
            self.camera = worker
        except Exception as exc:       # e.g. OpenCV missing
            self.camera = None
            self.mouse_mode = True
            self.engine.toast(f"Camera unavailable: {exc}", 5.0)

    def _poll_camera(self) -> None:
        cam = self.camera
        if cam is None:
            return
        self._cam_view.state = cam.state
        self._cam_view.message = cam.message
        self._cam_view.fps = cam.fps

        if cam.failed and not self.mouse_mode:
            self.mouse_mode = True
            self.engine.toast("Camera unavailable - mouse mode on", 4.0)

        obs = cam.latest()
        if obs.seq != self._seq:
            self._seq = obs.seq
            self._cam_view.hand = obs.landmarks is not None
            if obs.preview is not None:
                h, w = obs.preview.shape[:2]
                self._cam_view.preview = pygame.image.frombuffer(obs.preview.tobytes(), (w, h), "RGB")
            if not self.mouse_mode:
                self.engine.feed_hand(obs.landmarks, obs.t, obs.aspect)

    def _feed_mouse(self, now: float) -> None:
        pos = pygame.mouse.get_pos()
        left, _middle, right = pygame.mouse.get_pressed(3)
        if not pygame.mouse.get_focused():
            return
        gesture = Gesture.PALM if right else Gesture.DRAW if left else Gesture.NONE
        self.engine.feed_pointer(pos, gesture, now)

    # ------------------------------------------------------------------
    def _handle_event(self, e) -> None:
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.KEYDOWN:
            k = e.key
            if k == pygame.K_ESCAPE:
                self.running = False
            elif k == pygame.K_h:
                self.hud.show_help = not self.hud.show_help
                self._help_dismissed = True
            elif k == pygame.K_m:
                self.engine.toggle_mode()
            elif k in (pygame.K_e, pygame.K_SPACE):
                self.engine.cycle_effect()
            elif k == pygame.K_p:
                self.engine.toggle_pause()
            elif k == pygame.K_c:
                self.engine.clear()
            elif k in (pygame.K_f, pygame.K_F11):
                pygame.display.toggle_fullscreen()
            elif k == pygame.K_s:
                self._save_picture()
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            mode = self.hud.hit_toggle(e.pos)
            if mode is not None:
                self.engine.set_mode(mode)

    def _save_picture(self) -> None:
        folder = os.path.join(os.path.expanduser("~"), "Pictures")
        if not os.path.isdir(folder):
            folder = os.getcwd()
        path = os.path.join(folder, time.strftime("lumen_%Y%m%d_%H%M%S.png"))
        try:
            pygame.image.save(self.screen, path)
            self.engine.toast(f"Saved {os.path.basename(path)}", 3.0)
        except Exception as exc:
            self.engine.toast(f"Could not save picture: {exc}", 3.0)

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            while self.running:
                dt = self.clock.tick(C.TARGET_FPS) / 1000.0
                now = time.perf_counter()

                for e in pygame.event.get():
                    self._handle_event(e)

                self._poll_camera()
                if self.mouse_mode:
                    self._feed_mouse(now)

                self.engine.update(dt, now)

                # Fade the help card away once the person has started drawing.
                if self.hud.show_help and not self._help_dismissed and self.engine.has_drawn:
                    self.hud.show_help = False

                self.renderer.draw(self.screen, self.engine, dt, self.engine.t)
                self.hud.draw(self.screen, self.engine,
                              self._cam_view if self.camera is not None else None,
                              self.clock.get_fps(), self.mouse_mode)
                pygame.display.flip()
        finally:
            if self.camera is not None:
                self.camera.stop()
            pygame.quit()


def main(argv=None) -> None:
    args = parse_args(argv)
    LumenApp(args).run()
