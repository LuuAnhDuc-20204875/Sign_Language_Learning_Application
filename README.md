# Sign_Language_Learning_Application

Hand Tracking and Gesture Recognition for an **Interactive ASL Learning Application**.  
This project combines **real-time hand tracking**, **deep learning (MobileNetV2)** and a **PyQt5 desktop UI** to help users practice and learn the American Sign Language alphabet in a fun and gamified way.

---

## ✨ Features

- 🎥 **Real-time ASL Recognition**
  - Uses OpenCV, `cvzone` HandTrackingModule and a MobileNetV2 model.
  - Detects the hand, crops the ROI and predicts the ASL letter (A–Z + `del`, `nothing`, `space`).

- 📚 **Learn A–Z Mode**
  - Step-by-step practice for all letters from **A to Z**.
  - Shows a **reference image** for each sign (`EXAMPLE_IMG/Sample_A.jpeg`, …).
  - Requires a stable prediction over multiple frames before marking a letter as “correct”.

- 🎯 **Quiz Mode**
  - Randomly chooses target letters.
  - User signs the requested letter to gain score.
  - Great for self-testing and evaluation.

- 🐍 **Snake Game (Hand-Controlled)**
  - Classic snake game where the **index finger** position controls the snake.
  - Helps improve hand stability and coordination while keeping practice fun.

- 🔤 **Spelling Mode**
  - Practice spelling common words such as `HELLO`, `YES`, `NO`, `LOVE`, `NAME`.
  - User signs each letter in sequence; the app tracks progress and word accuracy.

- 👨‍🏫 **Teacher Mode**
  - Extra overlays for demonstration and debugging:
    - Live **FPS**.
    - **Confidence bar** for the model prediction.
  - Perfect for presentations or explaining how the system works.

- 📊 **Progress Tracking**
  - Saves statistics in `progress.json`:
    - Per-letter total attempts and correct attempts.
    - Per-word statistics in Spelling mode.
  - “Weak letters” (lowest accuracy) are highlighted in the UI to suggest what to practice more.

- ⚙️ **User Settings**
  - Camera flip (mirror mode) stored using `QSettings`.
  - Progress and configuration persist between runs.

---

## 🧱 Project Structure

```text
Sign_Language_Learning_Application/
├── EXAMPLE_IMG/          # Reference images: Sample_A.jpeg, Sample_B.jpeg, ...
├── cvzone/               # cvzone package (if vendored)
├── Donut.png             # Food sprite for Snake game
├── asl_app_pro.py        # Main application (PyQt5 + ASL logic)
├── progress.json         # Auto-generated progress log
├── requirements.txt      # Python dependencies
└── README.md
