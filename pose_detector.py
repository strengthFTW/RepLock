"""
pose_detector.py — MediaPipe PoseLandmarker wrapper (Tasks API, mediapipe ≥ 0.10.30)

Compatible with mediapipe 0.10.30+ which replaced mp.solutions with mp.tasks.
Requires: models/pose_landmarker_lite.task  (downloaded automatically on first run)
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Model auto-download ────────────────────────────────────────────────────────
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print("[INFO] Downloading pose landmarker model (5 MB) …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Model saved to", _MODEL_PATH)


# ── Landmark connections for drawing ─────────────────────────────────────────
# BlazePose 33-point connections (subset — enough for push-up skeleton)
_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (24, 26), (25, 27), (26, 28),              # legs
]


class PoseDetector:
    """Thin wrapper around mediapipe.tasks.python.vision.PoseLandmarker."""

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence:  float = 0.6,
        num_poses: int = 1,
    ):
        _ensure_model()

        base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts_ms = 0    # monotonically increasing timestamp (VIDEO mode)

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, frame):
        """
        Run pose estimation on a BGR OpenCV frame.

        Returns:
            PoseLandmarkerResult  (result.pose_landmarks is a list of lists)
            Access the first person's landmarks via result.pose_landmarks[0]
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33          # ~30 fps synthetic timestamp
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        return result

    def draw_landmarks(self, frame, result):
        """
        Draw the pose skeleton on the frame in-place.

        Args:
            frame:  BGR numpy array
            result: PoseLandmarkerResult from self.process()

        Returns:
            The same (mutated) frame.
        """
        if not result.pose_landmarks:
            return frame

        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks[0]   # first detected person

        # Draw connections
        for start_idx, end_idx in _POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            s = landmarks[start_idx]
            e = landmarks[end_idx]
            if s.visibility < 0.3 or e.visibility < 0.3:
                continue
            sx, sy = int(s.x * w), int(s.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            cv2.line(frame, (sx, sy), (ex, ey), (0, 200, 120), 2, cv2.LINE_AA)

        # Draw joints
        for lm in landmarks:
            if lm.visibility < 0.3:
                continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (0, 200, 120),   1,  cv2.LINE_AA)

        return frame

    def close(self):
        self._landmarker.close()
