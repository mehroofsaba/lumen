"""LUMEN's heads-up display: quiet, polished controls for a midnight canvas."""

import pygame

from . import config as C
from .effects import COLOR_TABLE
from .engine import HandMode, Mode
from .gestures import GESTURE_LABELS, Gesture

TEXT = (232, 235, 245)
DIM = (137, 146, 166)
MUTED = (91, 101, 123)
PANEL = (7, 10, 19)
GOLD = (244, 211, 126)
GOOD = (139, 220, 174)
WARN = (244, 199, 126)
BAD = (239, 132, 145)
BORDER = (104, 121, 151, 72)


def _font(size: int, bold: bool = False):
    try:
        return pygame.font.SysFont(
            "Aptos,Segoe UI Variable,Segoe UI,Arial,DejaVu Sans",
            size,
            bold=bold,
        )
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
        ("Index finger", "draw through the air"),
        ("Pinch", "pause / resume drawing"),
        ("Open palm", "scatter fairy dust"),
        ("Fist · hold", "clear the canvas"),
        ("Two fingers", "change particle effect"),
    )
    KEYS = "M style   E effect   P pause   C clear   S save   F fullscreen   H help   Esc quit"

    def __init__(self, size):
        self.w, self.h = size
        self.f_title = _font(54, True)
        self.f_big = _font(28, True)
        self.f_body = _font(18)
        self.f_small = _font(15)
        self.f_tiny = _font(12)
        self._text = {}
        self._panels = {}
        self.show_help = True
        self.toggle_rects = {}
        self.hand_toggle_rects = {}

    def _render(self, font, s: str, color):
        key = (id(font), s, color)
        surf = self._text.get(key)
        if surf is None:
            if len(self._text) > 500:
                self._text.clear()
            surf = font.render(s, True, color)
            self._text[key] = surf
        return surf

    def text(self, screen, font, s, pos, color=TEXT, anchor="topleft", shadow=False):
        surf = self._render(font, s, color)
        rect = surf.get_rect(**{anchor: pos})
        if shadow:
            screen.blit(self._render(font, s, (2, 3, 9)), rect.move(1, 2))
        screen.blit(surf, rect)
        return rect

    def panel(self, screen, rect, alpha=190, radius=18, border=BORDER):
        key = (rect.w, rect.h, alpha, radius, border)
        surf = self._panels.get(key)
        if surf is None:
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, PANEL + (alpha,), surf.get_rect(), border_radius=radius)
            if border:
                pygame.draw.rect(surf, border, surf.get_rect(), width=1, border_radius=radius)
            self._panels[key] = surf
        screen.blit(surf, rect.topleft)

    def hit_toggle(self, pos):
        for mode, rect in self.toggle_rects.items():
            if rect.collidepoint(pos):
                return mode
        return None

    def hit_hand_toggle(self, pos):
        for mode, rect in self.hand_toggle_rects.items():
            if rect.collidepoint(pos):
                return mode
        return None

    def draw(self, screen, engine, cam, fps: float, mouse_mode: bool) -> None:
        w, h = self.w, self.h
        fx = engine.effect
        accent = COLOR_TABLE[fx.color_ids[0]]

        # Brand block
        self.text(screen, self.f_title, "LUMEN", (34, 22), TEXT)
        pygame.draw.line(screen, (75, 86, 111), (38, 91), (78, 91), 2)
        self.text(screen, self.f_small, "AN INTERACTIVE FAIRY DUST EXPERIENCE",
                  (38, 101), DIM)
        self.text(screen, self.f_tiny, "A LITTLE SPACE TO MAKE LIGHT",
                  (38, 126), MUTED)

        # Drawing style selector
        seg_w, seg_h = 132, 38
        x0 = w - 36 - seg_w * 2
        self.text(screen, self.f_tiny, "DRAWING STYLE", (x0, 18), MUTED)
        self.panel(screen, pygame.Rect(x0 - 5, 35, seg_w * 2 + 10, seg_h + 8),
                   alpha=178, radius=23)
        self.toggle_rects.clear()
        for i, (mode, label) in enumerate(
            ((Mode.PERMANENT, "Permanent"), (Mode.FLEETING, "Fleeting"))
        ):
            rect = pygame.Rect(x0 + i * seg_w, 39, seg_w, seg_h)
            self.toggle_rects[mode] = rect
            if engine.mode == mode:
                pygame.draw.rect(screen, accent, rect, border_radius=19)
                self.text(screen, self.f_body, label, rect.center,
                          (17, 19, 25), "center")
            else:
                self.text(screen, self.f_body, label, rect.center, DIM, "center")
        self.text(screen, self.f_tiny, "M  TO SWITCH",
                  (w - 40, 86), MUTED, "topright")

        # Hand tracking selector
        hand_seg_w, hand_seg_h = 116, 34
        hand_x0 = w - 36 - hand_seg_w * 2
        self.text(screen, self.f_tiny, "HAND TRACKING", (hand_x0, 105), MUTED)
        self.panel(screen, pygame.Rect(hand_x0 - 5, 122, hand_seg_w * 2 + 10,
                                        hand_seg_h + 8), alpha=178, radius=21)
        self.hand_toggle_rects.clear()
        for i, (mode, label) in enumerate(
            ((HandMode.SINGLE, "Single"), (HandMode.DUAL, "Dual"))
        ):
            rect = pygame.Rect(hand_x0 + i * hand_seg_w, 126, hand_seg_w, hand_seg_h)
            self.hand_toggle_rects[mode] = rect
            if engine.hand_mode == mode:
                pygame.draw.rect(screen, accent, rect, border_radius=17)
                self.text(screen, self.f_body, label, rect.center,
                          (17, 19, 25), "center")
            else:
                self.text(screen, self.f_body, label, rect.center, DIM, "center")
        self.text(screen, self.f_tiny, "1  SINGLE     2  DUAL",
                  (w - 40, 170), MUTED, "topright")

        self._draw_effect(screen, engine, mouse_mode, accent)
        self._draw_camera(screen, cam, fps, mouse_mode)

        for i, (msg, left) in enumerate(reversed(engine.toasts)):
            alpha_hint = min(1.0, left / 0.5)
            col = tuple(int(c * alpha_hint) for c in TEXT)
            rect = pygame.Rect(0, 0, self.f_body.size(msg)[0] + 44, 40)
            rect.midbottom = (w // 2, h - 38 - i * 48)
            self.panel(screen, rect, alpha=int(205 * alpha_hint), radius=20)
            self.text(screen, self.f_body, msg, rect.center, col, "center")

        if self.show_help:
            self._draw_help(screen, mouse_mode)

    def _draw_effect(self, screen, engine, mouse_mode: bool, accent) -> None:
        fx = engine.effect
        box = pygame.Rect(28, self.h - 174, 430, 146)
        self.panel(screen, box, alpha=204, radius=20)
        self.text(screen, self.f_tiny, "CURRENT EFFECT",
                  (box.x + 20, box.y + 14), MUTED)
        self.text(screen, self.f_tiny, "TWO FINGERS  /  E",
                  (box.right - 20, box.y + 14), MUTED, "topright")
        self.text(screen, self.f_big, fx.name, (box.x + 20, box.y + 34), TEXT)
        self.text(screen, self.f_small, fx.blurb, (box.x + 22, box.y + 70), DIM)

        for i, cid in enumerate(fx.color_ids[:12]):
            pygame.draw.circle(screen, COLOR_TABLE[cid],
                               (box.x + 27 + i * 19, box.y + 105), 5)
        pygame.draw.line(screen, (48, 58, 77),
                         (box.x + 20, box.y + 119), (box.right - 20, box.y + 119), 1)
        self.text(screen, self.f_tiny, self._status_text(engine, mouse_mode),
                  (box.x + 20, box.y + 126), accent)

    def _status_text(self, engine, mouse_mode: bool) -> str:
        if not engine.hand_visible:
            return "Mouse ready · left draw · right scatter" if mouse_mode else "Waiting for your hand..."
        if engine.is_paused:
            return "Paused · pinch again to resume" if C.PINCH_MODE == "toggle" else "Paused · keep pinching"
        if engine.gesture == Gesture.FIST:
            return "Hold to clear the canvas..."
        label = GESTURE_LABELS.get(engine.gesture, "")
        return label or "Ready when you are"

    def _draw_camera(self, screen, cam, fps: float, mouse_mode: bool) -> None:
        pw, ph = C.PREVIEW_SIZE
        box = pygame.Rect(self.w - 28 - (pw + 24),
                          self.h - 28 - (ph + 92), pw + 24, ph + 92)
        self.panel(screen, box, alpha=204, radius=20)
        self.text(screen, self.f_tiny, "CAMERA", (box.x + 14, box.y + 13), MUTED)
        if cam is not None and cam.state == "running":
            state_label, state_color = "LIVE", GOOD
        elif cam is not None and cam.state == "error":
            state_label, state_color = "OFFLINE", BAD
        else:
            state_label, state_color = "STANDBY", WARN
        pygame.draw.circle(screen, state_color, (box.right - 54, box.y + 20), 3)
        self.text(screen, self.f_tiny, state_label,
                  (box.right - 44, box.y + 13), state_color)

        view = pygame.Rect(box.x + 12, box.y + 38, pw, ph)
        pygame.draw.rect(screen, (5, 8, 15), view, border_radius=7)
        if cam is not None and cam.preview is not None:
            screen.blit(cam.preview, view.topleft)
        pygame.draw.rect(screen, (91, 108, 137), view, 1, border_radius=7)

        if cam is None:
            color, msg = WARN, "Mouse mode · no camera"
        elif cam.state == "running":
            color, msg = (GOOD, "Hand detected") if cam.hand else (WARN, "Show your hand to begin")
        elif cam.state == "error":
            color, msg = BAD, cam.message
        else:
            color, msg = WARN, cam.message

        y = view.bottom + 10
        pygame.draw.circle(screen, color, (box.x + 19, y + 7), 4)
        for i, line in enumerate(wrap(self.f_small, msg, pw - 34)[:2]):
            self.text(screen, self.f_small, line, (box.x + 30, y + i * 18), TEXT)
        if cam is not None and cam.state == "running":
            self.text(screen, self.f_tiny,
                      f"CAM {cam.fps:0.0f}  /  APP {fps:0.0f} FPS",
                      (box.x + 12, box.bottom - 17), MUTED)

    def _draw_help(self, screen, mouse_mode: bool) -> None:
        card = pygame.Rect(0, 0, 570, 390)
        card.center = (self.w // 2, self.h // 2 - 8)
        self.panel(screen, card, alpha=232, radius=24,
                   border=(126, 143, 174, 105))
        self.text(screen, self.f_tiny, "QUICK GUIDE",
                  (card.centerx, card.y + 25), MUTED, "midtop")
        self.text(screen, self.f_big, "Make a little magic",
                  (card.centerx, card.y + 48), TEXT, "midtop")
        y = card.y + 108
        for gesture, action in self.HELP:
            self.text(screen, self.f_body, gesture, (card.x + 42, y), GOLD)
            self.text(screen, self.f_body, action, (card.x + 236, y), TEXT)
            y += 36
        if mouse_mode:
            self.text(screen, self.f_small,
                      "Mouse mode · left button draws · right button scatters",
                      (card.centerx, y + 2), WARN, "midtop")
        self.text(screen, self.f_tiny, self.KEYS,
                  (card.centerx, card.bottom - 62), DIM, "midtop")
        self.text(screen, self.f_small, "Press H to close this guide",
                  (card.centerx, card.bottom - 34), MUTED, "midtop")
