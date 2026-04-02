"""
ui.py — Overlay drawing helpers.

Renders a semi-transparent stats panel onto the camera frame with:
  • Rep counter
  • Stage indicator (UP / DOWN)
  • Elbow angle
  • Form feedback (colour-coded)
  • Keyboard shortcut hints
"""

import cv2
import numpy as np


# ── Colour palette (BGR) ──────────────────────────────────────────────────────
CLR_PANEL_BG    = (20,  20,  20)    # near-black panel background
CLR_ACCENT      = (0,  200, 120)    # green accent line / good feedback
CLR_BAD         = (0,   80, 220)    # red-ish (BGR) for bad feedback
CLR_LABEL       = (160, 160, 160)   # muted grey for labels
CLR_VALUE       = (255, 255, 255)   # white for values
CLR_STAGE_DOWN  = (0,  140, 255)    # orange for DOWN stage
CLR_STAGE_UP    = (0,  210, 100)    # green  for UP stage
CLR_HINT        = (100, 100, 100)   # dark grey for key hints


# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    frame: np.ndarray,
    count: int,
    stage: str,
    angle: float,
    feedback_msg: str,
    feedback_good: bool,
) -> np.ndarray:
    """
    Draw the full stats panel on the frame (non-destructive copy NOT made —
    frame is mutated in place for performance).

    Args:
        frame:        BGR numpy array from OpenCV
        count:        current rep count
        stage:        "up" or "down"
        angle:        current elbow angle (degrees)
        feedback_msg: string returned by give_feedback()
        feedback_good: True → green text, False → red text

    Returns:
        The mutated frame (same object).
    """
    h, w = frame.shape[:2]

    # ── Semi-transparent panel (left side) ───────────────────────────────────
    panel_x1, panel_y1 = 0,   0
    panel_x2, panel_y2 = 240, h

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2),
                  CLR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Accent line on right edge of panel
    cv2.line(frame, (panel_x2, 0), (panel_x2, h), CLR_ACCENT, 2)

    # ── Reps ─────────────────────────────────────────────────────────────────
    _label(frame, "REPS",  (20, 55))
    _value(frame, str(count), (20, 115), scale=2.4)

    # ── Stage ────────────────────────────────────────────────────────────────
    _label(frame, "STAGE", (20, 155))
    stage_clr = CLR_STAGE_DOWN if stage == "down" else CLR_STAGE_UP
    cv2.putText(frame, stage.upper(), (20, 205),
                FONT, 1.1, stage_clr, 2, cv2.LINE_AA)

    # ── Angle ────────────────────────────────────────────────────────────────
    _label(frame, "ANGLE", (20, 250))
    cv2.putText(frame, f"{angle:.0f} deg", (20, 295),
                FONT_SMALL, 0.85, CLR_VALUE, 1, cv2.LINE_AA)

    # ── Feedback ─────────────────────────────────────────────────────────────
    fb_clr = CLR_ACCENT if feedback_good else CLR_BAD
    _draw_feedback_bar(frame, feedback_msg, fb_clr, w, h)

    # ── Key hints ────────────────────────────────────────────────────────────
    cv2.putText(frame, "Q: quit  R: reset", (10, h - 12),
                FONT_SMALL, 0.45, CLR_HINT, 1, cv2.LINE_AA)

    return frame


def draw_angle_arc(frame: np.ndarray, vertex: tuple, angle: float):
    """
    Draw a small angle indicator arc at the elbow joint.

    Args:
        frame:  BGR frame
        vertex: (x, y) elbow pixel position
        angle:  elbow angle in degrees
    """
    radius = 30
    clr = CLR_ACCENT if angle < 100 else CLR_VALUE
    cv2.circle(frame, vertex, radius, clr, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{angle:.0f}", (vertex[0] + 35, vertex[1] + 10),
                FONT_SMALL, 0.6, clr, 1, cv2.LINE_AA)


# ── Private helpers ───────────────────────────────────────────────────────────

def _label(frame, text: str, pos: tuple):
    cv2.putText(frame, text, pos, FONT_SMALL, 0.55, CLR_LABEL, 1, cv2.LINE_AA)


def _value(frame, text: str, pos: tuple, scale: float = 1.4):
    cv2.putText(frame, text, pos, FONT, scale, CLR_VALUE, 3, cv2.LINE_AA)


def _draw_feedback_bar(frame, msg: str, colour: tuple, w: int, h: int):
    """Render a bottom feedback bar spanning the full width."""
    bar_h = 48
    bar_y = h - bar_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (240, bar_y), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (240, bar_y), (w, bar_y), colour, 2)

    # Centre the text in the bar
    (text_w, text_h), _ = cv2.getTextSize(msg, FONT_SMALL, 0.75, 2)
    text_x = 240 + ((w - 240) - text_w) // 2
    text_y = bar_y + (bar_h + text_h) // 2 - 4
    cv2.putText(frame, msg, (text_x, text_y),
                FONT_SMALL, 0.75, colour, 2, cv2.LINE_AA)
