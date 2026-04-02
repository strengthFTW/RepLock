# RepLock — AI Push-Up Counter And Form Tracker

A real-time, webcam-based push-up counter using **MediaPipe Pose**, **OpenCV**, and **NumPy**.

---

## Features

| Feature | Details |
|---|---|
| 🎯 Pose Detection | 33-keypoint skeleton via MediaPipe |
| 📐 Angle Calculation | Elbow angle averaged across both arms |
| 🔢 Rep Counting | Hysteresis state machine (85° / 160°) |
| 💬 Form Feedback | Depth · Hip Sag · Arm Symmetry checks |
| 🖥️ Live UI | Semi-transparent stats panel + feedback bar |

---

## Quick Start

### 1. Create & activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python main.py
```

Use a different camera (e.g., external USB):

```bash
python main.py --camera 1
```

---

## Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `R` | Reset counter |

---

## Project Structure

```
RepLock/
├── main.py           # Entry point
├── pose_detector.py  # MediaPipe Pose wrapper
├── counter.py        # Rep counting state machine
├── feedback.py       # Form feedback (rule-based)
├── ui.py             # Overlay drawing helpers
├── utils.py          # calculate_angle, extract_keypoints
└── requirements.txt
```

---

## How It Works

1. **Webcam frame** → mirrored (selfie view)
2. **MediaPipe** detects 33 body landmarks
3. **Elbow angle** computed from Shoulder → Elbow → Wrist
4. **State machine**: angle < 85° → DOWN; angle > 160° → UP + count++
5. **Form checks**: depth, hip sag, arm symmetry
6. **UI overlay**: stats panel (left) + feedback bar (bottom)

---

## Extending the Project

- **ML form classifier** — record rep data to CSV and train a scikit-learn model
- **Session history** — log reps + timestamps to SQLite
- **Web UI** — stream processed frames via FastAPI + WebSocket
