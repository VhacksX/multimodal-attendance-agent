# 🎓 EduSense AI — Multimodal Classroom Intelligence Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-attendance-agent.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/VhacksX/multimodal-attendance-agent)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)

**EduSense AI** is a real-time multimodal classroom monitoring and engagement telemetry agent. It synthesizes optical facial detection, visual attention classification, dynamic acoustic noise modeling, and live gamification into an executive dark-mode dashboard built with Streamlit and Plotly.

---

## 📌 Technical Terminology Glossary

* **Multimodal AI**: An artificial intelligence system capable of simultaneously ingesting, synchronizing, and reasoning over multiple disparate input streams (optical video frames combined with acoustic decibel telemetry) rather than relying on a single data sensor.
* **Haar Cascade Classifier**: A machine learning object-detection technique using cascade functions trained on positive and negative image patterns to rapidly detect visual structures (such as frontal faces and eye regions) with minimal computational latency.
* **Acoustic Telemetry**: The automated collection and measurement of sound parameters (such as root-mean-square decibel levels, or dB) from ambient surroundings to evaluate background environmental activity.
* **Edge Inference**: Running algorithmic processing and model evaluations directly on the local client machine, preserving bandwidth and preventing the transmission of private biometric video streams over external networks.

---

## 🚀 Key Features & Epics Completed

### 1. Optical Telemetry & Face Verification (EPIC 1 & 2)
* Real-time multi-face bounding box tracking via OpenCV cascades (`haarcascade_frontalface_default.xml`).
* Eye-tracking region of interest (ROI) classification distinguishing between **Attentive / Engaged** and **Distracted** student postures.
* Dynamic presence registration: initializes presence count strictly at `0` upon boot and dynamically registers additional students (`Student 1`, `Student 2`, `Student 3`) as they appear in the optical frame.

### 2. Calibrated Acoustic Context Analysis (EPIC 3)
* Real-time decibel monitoring tracking classroom ambient noise across calibrated pedagogical environments:
  * `-54.0 dB` to `-58.0 dB`: Quiet room standby (`quiet_ambient`)
  * `-40.0 dB` to `-45.0 dB`: Attentive individual work (`focused_study`)
  * `-28.0 dB` to `-35.0 dB`: Active classroom collaboration (`normal_discussion`)

### 3. Live Gamified Leaderboard (EPIC 4 & 5)
* Dynamic point attribution system starting strictly from `0 pts` at `00:00`.
* Real-time score increments calculated based on confirmed student engagement states.
* On-the-fly ranking and reordering of leaderboard rows dynamically rendered via isolated, format-safe HTML components.

### 4. Educator Central & Historical Analytics (EPIC 5)
* **Teacher Central**: Teaching effectiveness scorecard (87.5/100, +12% 30-day velocity), actionable recommendations, and historical session comparisons (CS101 vs. Math Workshop).
* **Session Analytics**: Plotly-powered timeline visualization graphing average attention trajectories and emotion frequency distributions.
* **Data Export**: Instant one-click CSV report export for offline attendance auditing.

### 5. Cloud-Adaptive Resilience (EPIC 6)
* Dual-mode runtime: automatically hooks into physical hardware cameras (`cv2.VideoCapture(0)`) when executed locally, and gracefully switches to a synthetic optical telemetry stream when deployed in headless cloud containers where physical webcams do not exist.
* Preconfigured with `packages.txt` (`libgl1`, `libglib2.0-0`) and `opencv-python-headless` for zero-configuration cloud container builds.

---

## 📂 Project Structure

```text
multimodal-attendance-agent/
├── backend/
│   └── core/
│       ├── face_detector.py         # Haar cascade face & eye attention classifier
│       └── face_recognition.py      # Presence verification & temporal thresholding
├── agent_module.py                  # Multimodal decision & heuristic aggregation logic
├── analytics.py                     # Historical engagement metrics & aggregation
├── audio_module.py                  # Acoustic decibel tracking & activity categorization
├── vision_module.py                 # Optical frame processing pipelines
├── app.py                           # Full-width Streamlit dashboard UI
├── requirements.txt                 # Headless Python dependencies
├── packages.txt                     # Debian Linux system dependencies (libgl1)
├── haarcascade_frontalface_default.xml
└── haarcascade_eye.xml
```

---

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VhacksX/multimodal-attendance-agent.git
   cd multimodal-attendance-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🌐 Live Deployment

* **Hosted Dashboard:** [https://multimodal-attendance-agent.streamlit.app/](https://multimodal-attendance-agent.streamlit.app/)
* **Repository Link:** [https://github.com/VhacksX/multimodal-attendance-agent](https://github.com/VhacksX/multimodal-attendance-agent)