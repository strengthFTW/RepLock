"""
counter.py — Push-up rep counting state machine.

Counting rules (all three must be satisfied):
  1. BOTH elbows must individually cross DOWN_THRESHOLD (85°) — no averaging tricks
  2. The DOWN phase must be held for at least DOWN_HOLD_MIN frames — filters out
     accidental arm bends and gestures
  3. BOTH elbows must individually cross UP_THRESHOLD (160°) to complete the rep

These three gates together eliminate:
  - False counts from single-arm movements (rule 1)
  - False counts from quick incidental flexions (rule 2)
  - Double counting at the boundary (hysteresis gap between 85° and 160°)
"""

DOWN_THRESHOLD = 85    # degrees — both elbows must dip below this
UP_THRESHOLD   = 160   # degrees — both elbows must rise above this
DOWN_HOLD_MIN  = 5     # frames — must stay in DOWN for this long to be "real"


class RepCounter:
    """Tracks push-up reps via a gated two-state machine (up / down)."""

    def __init__(self):
        self.count: int            = 0
        self.stage: str            = "up"
        self.min_angle_this_rep: float = 180.0
        self._down_hold: int       = 0    # consecutive frames spent in DOWN zone
        self._down_confirmed: bool = False  # True once hold minimum is met

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        left_angle:  float,
        right_angle: float,
    ) -> tuple[int, str]:
        """
        Feed BOTH elbow angles and receive updated (count, stage).

        Args:
            left_angle:  left  elbow angle in degrees
            right_angle: right elbow angle in degrees

        Returns:
            (count, stage) tuple
        """
        # Both arms must individually be below / above the threshold
        both_down = (left_angle < DOWN_THRESHOLD) and (right_angle < DOWN_THRESHOLD)
        both_up   = (left_angle > UP_THRESHOLD)   and (right_angle > UP_THRESHOLD)

        avg_angle = (left_angle + right_angle) / 2

        if both_down:
            self._down_hold += 1
            self.min_angle_this_rep = min(self.min_angle_this_rep, avg_angle)

            # Confirm the DOWN phase only after holding long enough
            if self._down_hold >= DOWN_HOLD_MIN:
                self._down_confirmed = True
                self.stage = "down"

        elif both_up and self._down_confirmed:
            # Full rep: confirmed DOWN → confirmed UP
            self.stage = "up"
            self.count += 1
            self.min_angle_this_rep = 180.0
            self._down_hold = 0
            self._down_confirmed = False

        else:
            # In-between zone — reset hold frame counter if not in DOWN zone
            if not both_down:
                self._down_hold = 0

        return self.count, self.stage

    def get_min_angle(self) -> float:
        """Return the deepest avg elbow angle reached in the current rep."""
        return self.min_angle_this_rep

    def reset(self):
        """Reset counter (bound to the 'r' key in main.py)."""
        self.count             = 0
        self.stage             = "up"
        self.min_angle_this_rep = 180.0
        self._down_hold        = 0
        self._down_confirmed   = False
