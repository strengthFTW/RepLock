"""
main.py — AI Push-Up Counter (RepLock MVP)

Controls:
    Q : quit
    R : reset rep counter

Run:
    python main.py
    python main.py --camera 1   # use a different camera index
"""

import argparse
import sys

import cv2

from pose_detector import PoseDetector
from utils         import calculate_angle, extract_keypoints
from counter       import RepCounter
from feedback      import give_feedback
from ui            import draw_overlay, draw_angle_arc


# ── BlazePose 33-point landmark indices ──────────────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24


def parse_args():
    p = argparse.ArgumentParser(description="RepLock — AI Push-Up Counter")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default 0 = built-in webcam)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Initialise ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = PoseDetector(
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    counter = RepCounter()

    # State shown on the UI when no person is in frame
    feedback_msg  = "Step into frame 🙏"
    feedback_good = True
    angle         = 0.0
    elbow_px      = (120, 300)

    print("[INFO] RepLock started.  Q → quit   R → reset counter")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Lost camera feed — retrying …")
            continue

        # Mirror for natural selfie-style view
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Pose detection ───────────────────────────────────────────────────
        result = detector.process(frame)
        detector.draw_landmarks(frame, result)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]  # first detected person

            # ── Extract joint coordinates ────────────────────────────────────
            r_joints = extract_keypoints(
                landmarks, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], w, h,
                visibility_threshold=0.6,  # both arms must be clearly visible
            )
            l_joints = extract_keypoints(
                landmarks, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], w, h,
                visibility_threshold=0.6,
            )
            hip_pts = extract_keypoints(
                landmarks, [RIGHT_HIP, LEFT_HIP], w, h,
                visibility_threshold=0.4,
            )

            if r_joints and l_joints:
                r_shoulder, r_elbow, r_wrist = r_joints
                l_shoulder, l_elbow, l_wrist = l_joints

                # ── Angle calculation ────────────────────────────────────────
                right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                left_angle  = calculate_angle(l_shoulder, l_elbow, l_wrist)

                angle    = (right_angle + left_angle) / 2  # display only
                elbow_px = r_elbow

                # ── Rep counting (BOTH arms must cross thresholds) ───────────
                count, stage = counter.update(
                    left_angle=left_angle,
                    right_angle=right_angle,
                )

                # ── Form feedback ────────────────────────────────────────────
                avg_hip_y = avg_shoulder_y = None
                if hip_pts:
                    avg_hip_y      = (hip_pts[0][1] + hip_pts[1][1]) // 2
                    avg_shoulder_y = (r_shoulder[1] + l_shoulder[1]) // 2

                fb = give_feedback(
                    stage              = stage,
                    min_angle_this_rep = counter.get_min_angle(),
                    hip_y_px           = avg_hip_y,
                    shoulder_y_px      = avg_shoulder_y,
                    left_elbow_angle   = left_angle,
                    right_elbow_angle  = right_angle,
                )
                feedback_msg  = fb.message
                feedback_good = fb.good

        # ── Draw UI ──────────────────────────────────────────────────────────
        draw_overlay(
            frame,
            count        = counter.count,
            stage        = counter.stage,
            angle        = angle,
            feedback_msg  = feedback_msg,
            feedback_good = feedback_good,
        )
        draw_angle_arc(frame, elbow_px, angle)

        cv2.imshow("RepLock — AI Push-Up Counter", frame)

        # ── Keyboard controls ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"[INFO] Session ended. Total reps: {counter.count}")
            break
        elif key == ord("r"):
            counter.reset()
            print("[INFO] Counter reset.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
