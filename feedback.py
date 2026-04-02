"""
feedback.py — Rule-based push-up form feedback.

Rules checked on every frame:
  1. Depth     — did the user go low enough at the bottom?
  2. Hip sag   — are hips dropping below shoulders?
  3. Symmetry  — are both elbows at a similar angle?
  4. Default   — everything looks good
"""

from dataclasses import dataclass


# ── Tunable thresholds ────────────────────────────────────────────────────────
DEPTH_MIN_ANGLE     = 100   # if bottom angle > this, user didn't go low enough
HIP_SAG_TOLERANCE   = 30    # pixels: hip can be this many px lower than shoulder
SYMMETRY_TOLERANCE  = 25    # degrees: difference between left/right elbow angles


# ── Feedback result ───────────────────────────────────────────────────────────
@dataclass
class FeedbackResult:
    message: str
    good: bool          # True → green, False → red/orange


# ── Public function ───────────────────────────────────────────────────────────
def give_feedback(
    stage: str,
    min_angle_this_rep: float,
    hip_y_px: int | None       = None,
    shoulder_y_px: int | None  = None,
    left_elbow_angle: float    = None,
    right_elbow_angle: float   = None,
) -> FeedbackResult:
    """
    Evaluate form using rule-based checks.

    Args:
        stage:               current rep stage ("up" / "down")
        min_angle_this_rep:  deepest elbow angle reached so far this rep
        hip_y_px:            y-pixel of the hip landmark (higher y = lower on screen)
        shoulder_y_px:       y-pixel of the shoulder landmark
        left_elbow_angle:    left elbow angle (degrees), if available
        right_elbow_angle:   right elbow angle (degrees), if available

    Returns:
        FeedbackResult with a message string and good/bad flag.
    """
    # 1. Depth check — only meaningful at the bottom of a rep
    if stage == "down" and min_angle_this_rep > DEPTH_MIN_ANGLE:
        return FeedbackResult("Go lower! 👇", good=False)

    # 2. Hip sag — hip y-coord much larger than shoulder y-coord (on screen = lower)
    if hip_y_px is not None and shoulder_y_px is not None:
        if hip_y_px > shoulder_y_px + HIP_SAG_TOLERANCE:
            return FeedbackResult("Don't sag your hips! ⚠️", good=False)

    # 3. Arm symmetry — only when both angles are available
    if left_elbow_angle is not None and right_elbow_angle is not None:
        if abs(left_elbow_angle - right_elbow_angle) > SYMMETRY_TOLERANCE:
            return FeedbackResult("Even out your arms! ↔️", good=False)

    # 4. Default — all checks passed
    return FeedbackResult("Great form! 💪", good=True)
