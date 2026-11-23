"""Heads-up display: title, mode toggle, effect info, camera panel, toasts, help."""

import pygame

from . import config as C
from .effects import COLOR_TABLE
from .engine import Mode
from .gestures import GESTURE_LABELS, Gesture

TEXT = (236, 230, 255)
DIM = (160, 152, 195)
PANEL = (14, 11, 34)
GOOD = (130, 235, 170)
WARN = (255, 205, 120)
BAD = (255, 130, 140)


def _font(size: int, bold: bool = False):
    try:
        return pygame.font.SysFont("segoeui,helveticaneue,helvetica,arial,dejavusans", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, int(size * 1.15))


def wrap(font, text: str, width: int):
    lines, line = [], ""
    for word in text.replace("\n", " \n ").split(" "):
        if word == "\n":
            lines.append(line)
            line = ""
            continue
        trial = (line + " " + word).strip()
        if font.size(trial)[0] <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class Hud:
    HELP = (
        ("Index finger", "draw"),
        ("Pinch", "pause / resume drawing"),
        ("Open palm", "scatter fairy dust"),
        ("Fist (hold)", "clear the canvas"),
        ("Two fingers", "change particle effect"),
    )
    KEYS = "M mode   E effect   P pause   C clear   S save picture   F fullscreen   H help   Esc quit"

    def __init__(self, size):
        self.w, self.h = size
        self.f_title = _font(46, True)
        self.f_big = _font(30, True)
        self.f_body = _font(20)
        self.f_small = _font(16)
        self.f_tiny = _font(14)
        self._text = {}
        self._panels = {}
        self.show_help = True
        self.toggle_rects = {}

    # ------------------------------------------------------------------
    def _render(self, font, s: str, color):
        key = (id(font), s, color)
        surf = self._text.get(key)
        if surf is None:
            if len(self._text) > 400:
                self._text.clear()
            surf = font.render(s, True, color)
            self._text[key] = surf
        return surf

    def text(self, screen, font, s, pos, color=TEXT, anchor="topleft", shadow=True):
        surf = self._render(font, s, color)
        rect = surf.get_rect(**{anchor: pos})
        if shadow:
            screen.blit(self._render(font, s, (4, 3, 14)), rect.move(1, 2))
        screen.blit(surf, rect)
        return rect

    def panel(self, screen, rect, alpha=150, radius=16, border=(120, 105, 190, 90)):
        key = (rect.w, rect.h, alpha, radius, border)
        surf = self._panels.get(key)
        if surf is None:
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, PANEL + (alpha,), surf.get_rect(), border_radius=radius)
            if border:
                pygame.draw.rect(surf, border, surf.get_rect(), width=1, border_radius=radius)
            self._panels[key] = surf
        screen.blit(surf, rect.topleft)

    # ------------------------------------------------------------------
    def hit_toggle(self, pos):
        """Returns the Mode whose on-screen toggle segment contains pos, else None."""
        for mode, rect in self.toggle_rects.items():
            if rect.collidepoint(pos):
                return mode
        return None

    # ------------------------------------------------------------------
    def draw(self, screen, engine, cam, fps: float, mouse_mode: bool) -> None:
        w, h = self.w, self.h
        fx = engine.effect
        accent = COLOR_TABLE[fx.color_ids[0]]

        # Title
        self.text(screen, self.f_title, "L U M E N", (36, 24), TEXT)
        self.text(screen, self.f_small, C.APP_TAGLINE, (40, 80), DIM)

        # Mode toggle (top right)
        seg_w, seg_h = 132, 38
        x0 = w - 36 - seg_w * 2
        self.panel(screen, pygame.Rect(x0 - 4, 24, seg_w * 2 + 8, seg_h + 8), alpha=150, radius=22)
        for i, (mode, label) in enumerate(((Mode.PERMANENT, "Permanent"), (Mode.FLEETING, "Fleeting"))):
            rect = pygame.Rect(x0 + i * seg_w, 28, seg_w, seg_h)
            self.toggle_rects[mode] = rect
            if engine.mode == mode:
                pygame.draw.rect(screen, accent, rect, border_radius=19)
                self.text(screen, self.f_body, label, rect.center, (24, 14, 44), "center", shadow=False)
            else:
                self.text(screen, self.f_body, label, rect.center, DIM, "center", shadow=False)
        self.text(screen, self.f_tiny, "drawing mode  (M)", (w - 36, 74), DIM, "topright")

        # Effect + gesture panel (bottom left)
        box = pygame.Rect(28, h - 150, 380, 122)
        self.panel(screen, box)
        self.text(screen, self.f_big, fx.name, (box.x + 20, box.y + 12), TEXT)
        self.text(screen, self.f_small, fx.blurb, (box.x + 22, box.y + 50), DIM)
        for i, cid in enumerate(fx.color_ids[:12]):
            pygame.draw.circle(screen, COLOR_TABLE[cid], (box.x + 28 + i * 18, box.y + 82), 6)
        self.text(screen, self.f_small, self._status_text(engine, mouse_mode), (box.x + 22, box.y + 96), accent)
        self.text(screen, self.f_tiny, "two fingers / E", (box.right - 16, box.y + 18), DIM, "topright")

        # Camera panel (bottom right)
        self._draw_camera(screen, cam, fps, mouse_mode)

        # Toasts
        for i, (msg, left) in enumerate(reversed(engine.toasts)):
            alpha_hint = min(1.0, left / 0.5)
            col = tuple(int(c * alpha_hint) for c in TEXT)
            rect = pygame.Rect(0, 0, self.f_body.size(msg)[0] + 40, 38)
            rect.midbottom = (w // 2, h - 40 - i * 46)
            self.panel(screen, rect, alpha=int(170 * alpha_hint), radius=19)
            self.text(screen, self.f_body, msg, rect.center, col, "center", shadow=False)

        if self.show_help:
            self._draw_help(screen, mouse_mode)

    # ------------------------------------------------------------------
    def _status_text(self, engine, mouse_mode: bool) -> str:
        if not engine.hand_visible:
            return "Mouse ready - left: draw, right: scatter" if mouse_mode else "Waiting for your hand..."
        if engine.is_paused:
            return "Paused - pinch again to resume" if C.PINCH_MODE == "toggle" else "Paused (pinching)"
        if engine.gesture == Gesture.FIST:
            return "Hold to clear..."
        label = GESTURE_LABELS.get(engine.gesture, "")
        return label or "Ready"

    def _draw_camera(self, screen, cam, fps: float, mouse_mode: bool) -> None:
        pw, ph = C.PREVIEW_SIZE
        box = pygame.Rect(self.w - 28 - (pw + 16), self.h - 28 - (ph + 76), pw + 16, ph + 76)
        self.panel(screen, box)
        self.text(screen, self.f_tiny, "CAMERA STATUS", (box.x + 12, box.y + 10), DIM)

        view = pygame.Rect(box.x + 8, box.y + 30, pw, ph)
        if cam is not None and cam.preview is not None:
            screen.blit(cam.preview, view.topleft)
        else:
            pygame.draw.rect(screen, (8, 6, 20), view)

        pygame.draw.rect(screen, (120, 105, 190), view, 1)

        if cam is None:
            color, msg = WARN, "Mouse mode (no camera)"
        elif cam.state == "running":
            color, msg = (GOOD, "Tracking your hand") if cam.hand else (WARN, "Show your hand to the camera")
        elif cam.state == "error":
            color, msg = BAD, cam.message
        else:
            color, msg = WARN, cam.message

        y = view.bottom + 8
        pygame.draw.circle(screen, color, (box.x + 18, y + 8), 5)
        for i, line in enumerate(wrap(self.f_small, msg, pw - 24)[:3]):
            self.text(screen, self.f_small, line, (box.x + 30, y + i * 18), TEXT)
        if cam is not None and cam.state == "running":
            self.text(screen, self.f_tiny, f"camera {cam.fps:0.0f} fps  |  app {fps:0.0f} fps",
                      (box.x + 12, box.bottom - 18), DIM)

    def _draw_help(self, screen, mouse_mode: bool) -> None:
        card = pygame.Rect(0, 0, 560, 380)
        card.center = (self.w // 2, self.h // 2 - 10)
        self.panel(screen, card, alpha=205, radius=22)
        self.text(screen, self.f_big, "Your controls", (card.centerx, card.y + 26), TEXT, "midtop")
        y = card.y + 84
        for gesture, action in self.HELP:
            self.text(screen, self.f_body, gesture, (card.x + 48, y), (255, 214, 140))
            self.text(screen, self.f_body, action, (card.x + 230, y), TEXT)
            y += 36
        if mouse_mode:
            self.text(screen, self.f_small, "Mouse mode: left button draws, right button scatters",
                      (card.centerx, y + 4), WARN, "midtop")
        self.text(screen, self.f_tiny, self.KEYS, (card.centerx, card.bottom - 62), DIM, "midtop")
        self.text(screen, self.f_small, "Press H to hide this panel", (card.centerx, card.bottom - 34), DIM, "midtop")
