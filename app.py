import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time

from backend.core.face_detector import FaceDetector
from backend.core.face_recognition import FaceRecognizer

st.set_page_config(
    page_title="EduSense AI - Classroom Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Full-Width Dark Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: #0c101a !important;
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stSidebar"] {
    display: none !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

div[data-testid="stTabs"] {
    margin-bottom: 18px;
}
div[data-testid="stTabs"] button {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
    padding: 8px 22px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2.5px solid #38bdf8 !important;
}

.edu-card {
    background-color: #141c2c;
    border: 1px solid #1e2a42;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.card-header-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    white-space: nowrap;
}

.brand-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.brand-subtitle {
    font-size: 0.78rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
}

.pill-session-active {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 9999px;
    margin-bottom: 14px;
}
.timer-text {
    font-size: 1.55rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.06em;
}
.stat-large-val {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin: 4px 0;
}

.blue-floating-banner {
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 18px;
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35);
    white-space: nowrap;
}
.banner-badge {
    background: rgba(255, 255, 255, 0.25);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.tag-engaged-green {
    background-color: #064e3b;
    color: #34d399;
    border: 1.5px solid #10b981;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.tag-distracted-amber {
    background-color: #78350f;
    color: #fcd34d;
    border: 1.5px solid #f59e0b;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.tag-waiting-slate {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1.5px solid #475569;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.badge-high-priority {
    background-color: #dc2626;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap;
}

div.stButton > button {
    background-color: #ef4444 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-size: 0.95rem !important;
}

.standby-screen {
    background: #141c2c;
    border: 1px solid #1e2a42;
    border-radius: 10px;
    height: 390px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# Single Top-Level Tab Container
tabs = st.tabs(["🖥️ Classroom Agent Feed", "👨‍🏫 Teacher Central", "📈 Session Analytics"])

# ==========================================
# 1. CLASSROOM AGENT FEED
# ==========================================
with tabs[0]:
    col_left, col_mid, col_right = st.columns([1.05, 2.7, 1.25], gap="medium")

    with col_left:
        st.markdown('<div class="brand-title">EduSense AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-subtitle">Classroom Intelligence</div>', unsafe_allow_html=True)

        st.markdown('<div class="pill-session-active">● SESSION ACTIVE</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='color:#94a3b8; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>SESSION DURATION</div>", unsafe_allow_html=True)
        timer_slot = st.empty()
        timer_slot.markdown("<div class='timer-text'>⏱ 00 : 00</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        students_slot = st.empty()
        students_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Students Present <span>👥</span></div>
<div class="stat-large-val">0</div>
</div>""", unsafe_allow_html=True)

        camera_toggle = st.toggle("Activate Live Camera Feed", value=False)
        st.button("⏹ End Session", use_container_width=True)

        st.markdown("""<div style="margin-top: 36px; color: #64748b; font-size: 0.85rem; font-weight: 600; cursor: pointer;">
⚙ Settings
</div>""", unsafe_allow_html=True)

    with col_mid:
        banner_slot = st.empty()
        banner_slot.markdown("""<div class="blue-floating-banner" style="background: linear-gradient(90deg, #334155 0%, #475569 100%); border-color: #64748b;">
<span class="banner-badge">STANDBY</span>
<span>Waiting for session activation...</span>
</div>""", unsafe_allow_html=True)

        video_slot = st.empty()
        video_slot.markdown("""<div class="standby-screen">
<div style="font-size: 2.4rem;">📷</div>
<div>Optical Camera Standby</div>
<div style="font-size: 0.82rem; color: #475569;">Toggle 'Activate Live Camera Feed' to start real-time telemetry</div>
</div>""", unsafe_allow_html=True)

        sub_c1, sub_c2 = st.columns([1.1, 1.3], gap="small")
        with sub_c1:
            leaderboard_slot = st.empty()
            leaderboard_slot.markdown("""<div class="edu-card">
<div class="card-header-label" style="color: #fbbf24;">
<span>🏆 LIVE LEADERBOARD</span>
<span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700;">● INITIALIZED</span>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; border-bottom: 1px solid #1e2a42; font-weight:600;">
<span>🟡 Student 1</span>
<span style="color: #38bdf8; font-weight: 800;">0 pts</span>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; font-weight:600;">
<span style="color: #cbd5e1;">⚪ Student 2</span>
<span style="color: #94a3b8; font-weight: 800;">0 pts</span>
</div>
</div>""", unsafe_allow_html=True)

        with sub_c2:
            suggestion_slot = st.empty()
            suggestion_slot.markdown("""<div class="edu-card">
<div class="card-header-label" style="color: #f43f5e;">
<span>💡 AI SUGGESTION BOX</span>
<span class="badge-high-priority">HIGH PRIORITY</span>
</div>
<div style="font-size: 0.98rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
Challenge students with harder problem
</div>
<div style="font-size: 0.82rem; color: #10b981; font-weight: 700;">
✦ Impact: Deep Learning
</div>
</div>""", unsafe_allow_html=True)

    with col_right:
        emotion_slot = st.empty()
        emotion_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Emotion Live</div>
<div style="margin-top: 10px;">
<span class="tag-waiting-slate">● idle</span>
</div>
</div>""", unsafe_allow_html=True)

        audio_slot = st.empty()
        audio_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Audio Context:</div>
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-top: 6px;">Noise Level</div>
<div class="stat-large-val" style="font-size: 2.5rem; margin: 2px 0 8px 0;">
-58.0 <span style="font-size: 1.0rem; color: #94a3b8;">dB</span>
</div>
<div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">
Activity: <span style="color: #38bdf8; font-weight: 700;">quiet_ambient</span>
</div>
</div>""", unsafe_allow_html=True)

        insights_slot = st.empty()
        insights_slot.markdown("""<div class="edu-card">
<div class="card-header-label">AI Insights</div>
<div style="font-size: 0.88rem; color: #94a3b8; font-weight: 500; line-height: 1.5; margin-top: 8px;">
Class session is ready. Activate live feed to initiate multimodal attention and attendance telemetry.
</div>
</div>""", unsafe_allow_html=True)

    # ---------------- DYNAMIC TELEMETRY LOOP ----------------
    if camera_toggle:
        try:
            detector = FaceDetector()
        except Exception:
            detector = None

        cap = cv2.VideoCapture(0)
        has_hardware_cam = (cap is not None and cap.isOpened())

        start_time = time.time()
        leaderboard_scores = {
            "Student 1": 0,
            "Student 2": 0
        }
        tick_counter = 0

        try:
            while camera_toggle:
                tick_counter += 1
                elapsed = int(time.time() - start_time)
                mins = elapsed // 60
                secs = elapsed % 60
                timer_slot.markdown(f"<div class='timer-text'>⏱ {mins:02d} : {secs:02d}</div>", unsafe_allow_html=True)

                if has_hardware_cam:
                    ret, frame = cap.read()
                    if not ret:
                        has_hardware_cam = False
                        continue
                    frame = cv2.flip(frame, 1)
                    faces = detector.detect_faces(frame) if detector else []
                else:
                    # Cloud Demo Mode: Generate synthetic dynamic optical stream
                    frame = np.full((390, 640, 3), (20, 28, 44), dtype=np.uint8)
                    cv2.putText(frame, "CLOUD DEMO STREAM - ATTENTION TELEMETRY", (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (cat << 'EOF' > app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time

from backend.core.face_detector import FaceDetector
from backend.core.face_recognition import FaceRecognizer

st.set_page_config(
    page_title="EduSense AI - Classroom Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Full-Width Dark Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: #0c101a !important;
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stSidebar"] {
    display: none !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

div[data-testid="stTabs"] {
    margin-bottom: 18px;
}
div[data-testid="stTabs"] button {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
    padding: 8px 22px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2.5px solid #38bdf8 !important;
}

.edu-card {
    background-color: #141c2c;
    border: 1px solid #1e2a42;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.card-header-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    white-space: nowrap;
}

.brand-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.brand-subtitle {
    font-size: 0.78rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
}

.pill-session-active {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 9999px;
    margin-bottom: 14px;
}
.timer-text {
    font-size: 1.55rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.06em;
}
.stat-large-val {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin: 4px 0;
}

.blue-floating-banner {
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 18px;
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35);
    white-space: nowrap;
}
.banner-badge {
    background: rgba(255, 255, 255, 0.25);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.tag-engaged-green {
    background-color: #064e3b;
    color: #34d399;
    border: 1.5px solid #10b981;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.tag-distracted-amber {
    background-color: #78350f;
    color: #fcd34d;
    border: 1.5px solid #f59e0b;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.tag-waiting-slate {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1.5px solid #475569;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.badge-high-priority {
    background-color: #dc2626;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap;
}

div.stButton > button {
    background-color: #ef4444 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-size: 0.95rem !important;
}

.standby-screen {
    background: #141c2c;
    border: 1px solid #1e2a42;
    border-radius: 10px;
    height: 390px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# Single Top-Level Tab Container
tabs = st.tabs(["🖥️ Classroom Agent Feed", "👨‍🏫 Teacher Central", "📈 Session Analytics"])

# ==========================================
# 1. CLASSROOM AGENT FEED
# ==========================================
with tabs[0]:
    col_left, col_mid, col_right = st.columns([1.05, 2.7, 1.25], gap="medium")

    with col_left:
        st.markdown('<div class="brand-title">EduSense AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-subtitle">Classroom Intelligence</div>', unsafe_allow_html=True)

        st.markdown('<div class="pill-session-active">● SESSION ACTIVE</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='color:#94a3b8; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>SESSION DURATION</div>", unsafe_allow_html=True)
        timer_slot = st.empty()
        timer_slot.markdown("<div class='timer-text'>⏱ 00 : 00</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        students_slot = st.empty()
        students_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Students Present <span>👥</span></div>
<div class="stat-large-val">0</div>
</div>""", unsafe_allow_html=True)

        camera_toggle = st.toggle("Activate Live Camera Feed", value=False)
        st.button("⏹ End Session", use_container_width=True)

        st.markdown("""<div style="margin-top: 36px; color: #64748b; font-size: 0.85rem; font-weight: 600; cursor: pointer;">
⚙ Settings
</div>""", unsafe_allow_html=True)

    with col_mid:
        banner_slot = st.empty()
        banner_slot.markdown("""<div class="blue-floating-banner" style="background: linear-gradient(90deg, #334155 0%, #475569 100%); border-color: #64748b;">
<span class="banner-badge">STANDBY</span>
<span>Waiting for session activation...</span>
</div>""", unsafe_allow_html=True)

        video_slot = st.empty()
        video_slot.markdown("""<div class="standby-screen">
<div style="font-size: 2.4rem;">📷</div>
<div>Optical Camera Standby</div>
<div style="font-size: 0.82rem; color: #475569;">Toggle 'Activate Live Camera Feed' to start real-time telemetry</div>
</div>""", unsafe_allow_html=True)

        sub_c1, sub_c2 = st.columns([1.1, 1.3], gap="small")
        with sub_c1:
            leaderboard_slot = st.empty()
            leaderboard_slot.markdown("""<div class="edu-card">
<div class="card-header-label" style="color: #fbbf24;">
<span>🏆 LIVE LEADERBOARD</span>
<span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700;">● INITIALIZED</span>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; border-bottom: 1px solid #1e2a42; font-weight:600;">
<span>🟡 Student 1</span>
<span style="color: #38bdf8; font-weight: 800;">0 pts</span>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; font-weight:600;">
<span style="color: #cbd5e1;">⚪ Student 2</span>
<span style="color: #94a3b8; font-weight: 800;">0 pts</span>
</div>
</div>""", unsafe_allow_html=True)

        with sub_c2:
            suggestion_slot = st.empty()
            suggestion_slot.markdown("""<div class="edu-card">
<div class="card-header-label" style="color: #f43f5e;">
<span>💡 AI SUGGESTION BOX</span>
<span class="badge-high-priority">HIGH PRIORITY</span>
</div>
<div style="font-size: 0.98rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
Challenge students with harder problem
</div>
<div style="font-size: 0.82rem; color: #10b981; font-weight: 700;">
✦ Impact: Deep Learning
</div>
</div>""", unsafe_allow_html=True)

    with col_right:
        emotion_slot = st.empty()
        emotion_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Emotion Live</div>
<div style="margin-top: 10px;">
<span class="tag-waiting-slate">● idle</span>
</div>
</div>""", unsafe_allow_html=True)

        audio_slot = st.empty()
        audio_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Audio Context:</div>
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-top: 6px;">Noise Level</div>
<div class="stat-large-val" style="font-size: 2.5rem; margin: 2px 0 8px 0;">
-58.0 <span style="font-size: 1.0rem; color: #94a3b8;">dB</span>
</div>
<div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">
Activity: <span style="color: #38bdf8; font-weight: 700;">quiet_ambient</span>
</div>
</div>""", unsafe_allow_html=True)

        insights_slot = st.empty()
        insights_slot.markdown("""<div class="edu-card">
<div class="card-header-label">AI Insights</div>
<div style="font-size: 0.88rem; color: #94a3b8; font-weight: 500; line-height: 1.5; margin-top: 8px;">
Class session is ready. Activate live feed to initiate multimodal attention and attendance telemetry.
</div>
</div>""", unsafe_allow_html=True)

    # ---------------- DYNAMIC TELEMETRY LOOP ----------------
    if camera_toggle:
        try:
            detector = FaceDetector()
        except Exception:
            detector = None

        cap = cv2.VideoCapture(0)
        has_hardware_cam = (cap is not None and cap.isOpened())

        start_time = time.time()
        leaderboard_scores = {
            "Student 1": 0,
            "Student 2": 0
        }
        tick_counter = 0

        try:
            while camera_toggle:
                tick_counter += 1
                elapsed = int(time.time() - start_time)
                mins = elapsed // 60
                secs = elapsed % 60
                timer_slot.markdown(f"<div class='timer-text'>⏱ {mins:02d} : {secs:02d}</div>", unsafe_allow_html=True)

                if has_hardware_cam:
                    ret, frame = cap.read()
                    if not ret:
                        has_hardware_cam = False
                        continue
                    frame = cv2.flip(frame, 1)
                    faces = detector.detect_faces(frame) if detector else []
                else:
                    # Cloud Demo Mode: Generate synthetic dynamic optical stream
                    frame = np.full((390, 640, 3), (20, 28, 44), dtype=np.uint8)
                    cv2.putText(frame, "CLOUD DEMO STREAM - ATTENTION TELEMETRY", (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 2)
                    # Simulated face location
                    faces = [(220, 100, 200, 200)]

                num_faces = len(faces)
                students_slot.markdown(f"""<div class="edu-card">
<div class="card-header-label">Students Present <span>👥</span></div>
<div class="stat-large-val">{num_faces}</div>
</div>""", unsafe_allow_html=True)

                # Draw LIVE indicator
                cv2.rectangle(frame, (16, 14), (88, 44), (15, 30, 20), -1)
                cv2.circle(frame, (30, 29), 5, (0, 255, 0), -1)
                cv2.putText(frame, "LIVE", (44, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                student_engagement_states = {}

                if num_faces == 0:
                    emotion_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Emotion Live</div>
<div style="margin-top: 10px;">
<span class="tag-waiting-slate">● waiting</span>
</div>
</div>""", unsafe_allow_html=True)

                    banner_slot.markdown("""<div class="blue-floating-banner" style="background: linear-gradient(90deg, #334155 0%, #475569 100%); border-color: #64748b;">
<span class="banner-badge">STANDBY</span>
<span>Waiting for student detection in optical frame...</span>
</div>""", unsafe_allow_html=True)
                else:
                    for idx, (x, y, w, h) in enumerate(faces):
                        if has_hardware_cam and detector:
                            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
                            attention = detector.classify_attention(roi_gray)
                            is_attentive = ("Attentive" in attention or attention == "engaged")
                        else:
                            is_attentive = True

                        status_str = "engaged" if is_attentive else "distracted"
                        student_name = f"Student {idx + 1}"
                        student_engagement_states[student_name] = is_attentive

                        if student_name not in leaderboard_scores:
                            leaderboard_scores[student_name] = 0

                        color = (0, 255, 0) if is_attentive else (0, 165, 255)
                        tag_label = f"ID: {idx + 1} | {status_str}"

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame, (x, y - 28), (x + len(tag_label) * 11, y), color, -1)
                        cv2.putText(frame, tag_label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    primary_engaged = student_engagement_states.get("Student 1", False)
                    if primary_engaged:
                        emotion_slot.markdown("""<div class="edu-card">
<div class="card-header-label">Emotion Live</div>
<div style="margin-top: 10px;">
<span class="tag-engaged-green">● engaged</span>
</div>
</div>""", unsafe_allow_html=True)

                        banner_slot.markdown("""<div class="blue-floating-banner">
<span class="banner-badge">SUCCESS</span>
<span>Engagement is optimal. Keep going!</span>
</div>""", unsafe_allow_html=True)

                if tick_counter % 12 == 0:
                    for s_name, engaged in student_engagement_states.items():
                        if engaged:
                            leaderboard_scores[s_name] += 1

                ranked_students = sorted(leaderboard_scores.items(), key=lambda item: item[1], reverse=True)

                rows_html = ""
                for rank_idx, (s_name, s_pts) in enumerate(ranked_students):
                    bullet = "🟡" if rank_idx == 0 else "⚪"
                    pt_color = "#38bdf8" if rank_idx == 0 else "#94a3b8"
                    border_style = "border-bottom: 1px solid #1e2a42;" if rank_idx < len(ranked_students) - 1 else ""
                    rows_html += f'<div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; {border_style} font-weight:600;"><span>{bullet} {s_name}</span><span style="color: {pt_color}; font-weight: 800;">{s_pts} pts</span></div>'

                leaderboard_slot.markdown(f"""<div class="edu-card">
<div class="card-header-label" style="color: #fbbf24;">
<span>🏆 LIVE LEADERBOARD</span>
<span style="font-size: 0.72rem; color: #34d399; font-weight: 700; background: rgba(52, 211, 153, 0.15); padding: 2px 7px; border-radius: 4px;">● LIVE</span>
</div>
{rows_html}
</div>""", unsafe_allow_html=True)

                base_db = -43.5 if num_faces > 0 else -54.0
                current_db = base_db + np.sin(time.time() * 2.5) * 1.2
                audio_slot.markdown(f"""<div class="edu-card">
<div class="card-header-label">Audio Context:</div>
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-top: 6px;">Noise Level</div>
<div class="stat-large-val" style="font-size: 2.5rem; margin: 2px 0 8px 0;">
{current_db:.1f} <span style="font-size: 1.0rem; color: #94a3b8;">dB</span>
</div>
<div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">
Activity: <span style="color: #38bdf8; font-weight: 700;">focused_study</span>
</div>
</div>""", unsafe_allow_html=True)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_slot.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.05)

        finally:
            if cap is not None:
                cap.release()

# ==========================================
# 2. TEACHER CENTRAL VIEW
# ==========================================
with tabs[1]:
    st.markdown('<div class="brand-title" style="margin-top: 10px;">Teacher Central</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8; font-size:1.0rem; font-weight:600; margin-top:-4px; margin-bottom: 24px;">Teacher Profile &mdash; Effectiveness Dashboard & Insights</p>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3, gap="medium")
    with t1:
        st.markdown("""<div class="edu-card">
<div class="card-header-label">TEACHING EFFECTIVENESS</div>
<div class="stat-large-val">87.5 <span style="font-size:1.2rem; color:#94a3b8;">/ 100</span></div>
<div style="color:#10b981; font-size:0.9rem; font-weight:700; margin-top:8px;">↗ +12% 30-day trend</div>
</div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("""<div class="edu-card">
<div class="card-header-label">STRENGTHS</div>
<ul style="margin-top:10px; padding-left:18px; font-size:0.95rem; color:#e2e8f0; line-height:1.6;">
<li>High engagement sessions (85%+)</li>
<li>Emotion diversity management</li>
</ul>
</div>""", unsafe_allow_html=True)

    with t3:
        st.markdown("""<div class="edu-card">
<div class="card-header-label">RECOMMENDATIONS</div>
<ul style="margin-top:10px; padding-left:18px; font-size:0.95rem; color:#e2e8f0; line-height:1.6;">
<li>Try interactive polls (+18%)</li>
<li>Short video breaks (+25% re-engagement)</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.markdown('<p style="font-size:1.1rem; font-weight:800; margin-top:18px; margin-bottom: 12px;">Class History Comparison</p>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3, gap="medium")
    with h1:
        st.markdown("""<div class="edu-card">
<div style="font-weight:700; font-size:1.05rem;">CS101 (Today)</div>
<div style="font-size:2.0rem; font-weight:800; color:#818cf8; margin-top:4px;">85% <span style="font-size:0.9rem; color:#94a3b8;">Avg</span></div>
</div>""", unsafe_allow_html=True)
    with h2:
        st.markdown("""<div class="edu-card">
<div style="font-weight:700; font-size:1.05rem;">CS101 (Yesterday)</div>
<div style="font-size:2.0rem; font-weight:800; color:#38bdf8; margin-top:4px;">78% <span style="font-size:0.9rem; color:#94a3b8;">Avg</span></div>
</div>""", unsafe_allow_html=True)
    with h3:
        st.markdown("""<div class="edu-card">
<div style="font-weight:700; font-size:1.05rem;">Math Workshop</div>
<div style="font-size:2.0rem; font-weight:800; color:#34d399; margin-top:4px;">92% <span style="font-size:0.9rem; color:#94a3b8;">Avg</span></div>
</div>""", unsafe_allow_html=True)

# ==========================================
# 3. SESSION ANALYTICS & REPORTS
# ==========================================
with tabs[2]:
    st.markdown('<div class="brand-title" style="margin-top: 10px;">Session Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8; font-size:1.0rem; font-weight:600; margin-top:-4px; margin-bottom: 24px;">Deep dive into classroom engagement timelines</p>', unsafe_allow_html=True)

    timestamps = [f"08:47:{s:02d}" for s in range(56, 60)] + [f"08:48:{s:02d}" for s in range(0, 50, 4)]
    attention_vals = [82, 84, 85, 87, 88, 52, 85, 86, 45, 88, 89, 87, 85, 90, 92, 88, 86]
    emotion_vals = [1.0, 1.2, 0.8, 1.4, 1.9, 1.1, 1.0, 1.2, 1.0, 1.1, 0.9, 1.4, 1.2, 1.0, 1.5, 1.1, 1.0]

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("##### Attention Trend")
        df_att = pd.DataFrame({"Time": timestamps[:len(attention_vals)], "avg_attention": attention_vals})
        fig1 = px.line(df_att, x="Time", y="avg_attention", template="plotly_dark")
        fig1.update_traces(line_color="#818cf8", line_width=3)
        fig1.update_layout(paper_bgcolor="#141c2c", plot_bgcolor="#141c2c", margin=dict(l=15, r=15, t=15, b=15))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("##### Emotion Distribution Over Time")
        df_emo = pd.DataFrame({"Time": timestamps[:len(emotion_vals)], "Count": emotion_vals})
        fig2 = px.bar(df_emo, x="Time", y="Count", template="plotly_dark", color_discrete_sequence=["#10b981"])
        fig2.update_layout(paper_bgcolor="#141c2c", plot_bgcolor="#141c2c", margin=dict(l=15, r=15, t=15, b=15))
        st.plotly_chart(fig2, use_container_width=True)

    csv_data = df_att.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Session CSV Report",
        data=csv_data,
        file_name="session_analytics.csv",
        mime="text/csv"
    )
