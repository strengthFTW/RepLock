"""
utils.py — Shared math utilities
  - calculate_angle(a, b, c): angle in degrees at vertex b
  - extract_keypoints(landmarks, indices, w, h): pixel (x, y) tuples
                                                  from Tasks API NormalizedLandmark list
"""

import numpy as np


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Compute the angle (in degrees) at joint b, given three (x, y) points.

    Args:
        a: (x, y) of the first joint  (e.g. shoulder)
        b: (x, y) of the vertex joint (e.g. elbow)  ← angle measured here
        c: (x, y) of the third joint  (e.g. wrist)

    Returns:
        Angle in degrees [0, 180].
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    angle = np.degrees(
        np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)


def extract_keypoints(
    landmarks: list,       # list[NormalizedLandmark] from Tasks API
    indices: list[int],
    frame_w: int,
    frame_h: int,
    visibility_threshold: float = 0.3,
) -> list[tuple] | None:
    """
    Pull (x, y) pixel coordinates for the given landmark indices.

    Args:
        landmarks:            list of NormalizedLandmark (result.pose_landmarks[0])
        indices:              landmark IDs to extract (e.g. [11, 13, 15])
        frame_w / frame_h:    frame dimensions in pixels
        visibility_threshold: skip extraction if any joint is below this confidence

    Returns:
        List of (x, y) pixel tuples, or None if any landmark is below threshold.
    """
    points = []
    for idx in indices:
        lm = landmarks[idx]
        if lm.visibility < visibility_threshold:
            return None          # signal to the caller to skip this frame
        points.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    return points
