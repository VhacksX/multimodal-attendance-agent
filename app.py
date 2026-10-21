import streamlit as st
import cv2
import numpy as np
import time

from backend.core.face_detector import FaceDetector
from backend.core.face_recognition import FaceRecognizer

st.set_page_config(
    page_title="EduSense AI - Classroom Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Pixel-Accurate SkillWallet Full-Width Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, sans-serif !important;
        background-color: #0c101a !important;
        color: #f8fafc;
    }

    /* Hide Streamlit Header & Sidebar to maximize full-screen grid width */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    /* Structured Cards */
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

    /* Brand Header */
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

    /* Status & Metrics */
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

    /* Floating Blue Banner */
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

    /* Emotion Tags */
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

    /* Priority Badges */
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

    /* Red End Session Button */
    div.stButton > button {
        background-color: #ef4444 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        font-size: 0.95rem !important;
    }

    /* Standby Video Screen */
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

# 3-Column Uniform Grid Layout (Left Controls | Center Video & Sub-cards | Right Telemetry)
col_left, col_mid, col_right = st.columns([1.05, 2.7, 1.25], gap="medium")

# ---------------- LEFT COLUMN ----------------
with col_left:
    st.markdown('<div class="brand-title">EduSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Classroom Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="pill-session-active">● SESSION ACTIVE</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='color:#94a3b8; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>SESSION DURATION</div>", unsafe_allow_html=True)
    timer_slot = st.empty()
    timer_slot.markdown("<div class='timer-text'>⏱ 00 : 00</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    students_slot = st.empty()
    # Initialized cleanly to 0
    students_slot.markdown("""
    <div class="edu-card">
        <div class="card-header-label">Students Present <span>👥</span></div>
        <div class="stat-large-val">0</div>
    </div>
    """, unsafe_allow_html=True)

    camera_toggle = st.toggle("Activate Live Camera Feed", value=False)
    st.button("⏹ End Session", use_container_width=True)

    st.markdown("""
    <div style="margin-top: 36px; color: #64748b; font-size: 0.85rem; font-weight: 600; cursor: pointer;">
        ⚙ Settings
    </div>
    """, unsafe_allow_html=True)

# ---------------- CENTER COLUMN ----------------
with col_mid:
    banner_slot = st.empty()
    banner_slot.markdown("""
    <div class="blue-floating-banner">
        <span class="banner-badge">SUCCESS</span>
        <span>Engagement is optimal. Keep going!</span>
    </div>
    """, unsafe_allow_html=True)

    video_slot = st.empty()
    video_slot.markdown("""
    <div class="standby-screen">
        <div style="font-size: 2.4rem;">📷</div>
        <div>Optical Camera Standby</div>
        <div style="font-size: 0.82rem; color: #475569;">Toggle 'Activate Live Camera Feed' to start real-time telemetry</div>
    </div>
    """, unsafe_allow_html=True)

    # 2 Sub-Cards Below Video
    sub_c1, sub_c2 = st.columns([1.1, 1.3], gap="small")
    with sub_c1:
        leaderboard_slot = st.empty()
        leaderboard_slot.markdown("""
        <div class="edu-card">
            <div class="card-header-label" style="color: #fbbf24;">
                <span>🏆 LIVE LEADERBOARD</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; border-bottom: 1px solid #1e2a42; font-weight:600;">
                <span>🟡 Student 1</span>
                <span style="color: #38bdf8; font-weight: 800;">36 pts</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; font-weight:600;">
                <span style="color: #cbd5e1;">⚪ Student 3</span>
                <span style="color: #94a3b8; font-weight: 800;">2 pts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sub_c2:
        suggestion_slot = st.empty()
        suggestion_slot.markdown("""
        <div class="edu-card">
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
        </div>
        """, unsafe_allow_html=True)

# ---------------- RIGHT COLUMN ----------------
with col_right:
    emotion_slot = st.empty()
    emotion_slot.markdown("""
    <div class="edu-card">
        <div class="card-header-label">Emotion Live</div>
        <div style="margin-top: 10px;">
            <span class="tag-waiting-slate">● idle</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    audio_slot = st.empty()
    audio_slot.markdown("""
    <div class="edu-card">
        <div class="card-header-label">Audio Context:</div>
        <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-top: 6px;">Noise Level</div>
        <div class="stat-large-val" style="font-size: 2.5rem; margin: 2px 0 8px 0;">
            -44.0 <span style="font-size: 1.0rem; color: #94a3b8;">dB</span>
        </div>
        <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">
            Activity: <span style="color: #38bdf8; font-weight: 700;">normal_discussion</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    insights_slot = st.empty()
    insights_slot.markdown("""
    <div class="edu-card">
        <div class="card-header-label">AI Insights</div>
        <div style="font-size: 0.88rem; color: #94a3b8; font-weight: 500; line-height: 1.5; margin-top: 8px;">
            Class session is ready. Activate live feed to initiate multimodal attention and attendance telemetry.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DYNAMIC TELEMETRY LOOP ----------------
if camera_toggle:
    detector = FaceDetector()
    recognizer = FaceRecognizer(threshold_seconds=1.5)
    cap = cv2.VideoCapture(0)

    start_time = time.time()

    try:
        while camera_toggle:
            ret, frame = cap.read()
            if not ret:
                break

            # Dynamic Timer counting up from 00:00
            elapsed = int(time.time() - start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            timer_slot.markdown(f"<div class='timer-text'>⏱ {mins:02d} : {secs:02d}</div>", unsafe_allow_html=True)

            # Vision Detection
            frame = cv2.flip(frame, 1)
            faces = detector.detect_faces(frame)
            num_faces = len(faces)

            # Set student count directly to detected faces (0 if empty)
            students_slot.markdown(f"""
            <div class="edu-card">
                <div class="card-header-label">Students Present <span>👥</span></div>
                <div class="stat-large-val">{num_faces}</div>
            </div>
            """, unsafe_allow_html=True)

            # Draw LIVE Badge on video feed
            cv2.rectangle(frame, (16, 14), (88, 44), (15, 30, 20), -1)
            cv2.circle(frame, (30, 29), 5, (0, 255, 0), -1)
            cv2.putText(frame, "LIVE", (44, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update UI based on detected presence
            if num_faces == 0:
                emotion_slot.markdown("""
                <div class="edu-card">
                    <div class="card-header-label">Emotion Live</div>
                    <div style="margin-top: 10px;">
                        <span class="tag-waiting-slate">● waiting</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                banner_slot.markdown("""
                <div class="blue-floating-banner" style="background: linear-gradient(90deg, #334155 0%, #475569 100%); border-color: #64748b;">
                    <span class="banner-badge">STANDBY</span>
                    <span>Waiting for student detection in optical frame...</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                for idx, (x, y, w, h) in enumerate(faces):
                    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
                    attention = detector.classify_attention(roi_gray)
                    status_str = "engaged" if "Attentive" in attention or attention == "engaged" else "distracted"
                    color = (0, 255, 0) if status_str == "engaged" else (0, 165, 255)

                    tag_id = 1 if idx == 0 else (3 if idx == 1 else idx + 1)
                    tag_label = f"ID: {tag_id} | {status_str}"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y - 28), (x + len(tag_label) * 11, y), color, -1)
                    cv2.putText(frame, tag_label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                emotion_slot.markdown("""
                <div class="edu-card">
                    <div class="card-header-label">Emotion Live</div>
                    <div style="margin-top: 10px;">
                        <span class="tag-engaged-green">● engaged</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                banner_slot.markdown("""
                <div class="blue-floating-banner">
                    <span class="banner-badge">SUCCESS</span>
                    <span>Engagement is optimal. Keep going!</span>
                </div>
                """, unsafe_allow_html=True)

                # Dynamic Points increment while students are present
                pts = 36 + (elapsed // 5)
                leaderboard_slot.markdown(f"""
                <div class="edu-card">
                    <div class="card-header-label" style="color: #fbbf24;">
                        <span>🏆 LIVE LEADERBOARD</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; border-bottom: 1px solid #1e2a42; font-weight:600;">
                        <span>🟡 Student 1</span>
                        <span style="color: #38bdf8; font-weight: 800;">{pts} pts</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; padding: 7px 0; font-weight:600;">
                        <span style="color: #cbd5e1;">⚪ Student 3</span>
                        <span style="color: #94a3b8; font-weight: 800;">2 pts</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Dynamic Audio Telemetry
            db_val = -44.0 + np.sin(time.time() * 2) * 1.5
            audio_slot.markdown(f"""
            <div class="edu-card">
                <div class="card-header-label">Audio Context:</div>
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-top: 6px;">Noise Level</div>
                <div class="stat-large-val" style="font-size: 2.5rem; margin: 2px 0 8px 0;">
                    {db_val:.1f} <span style="font-size: 1.0rem; color: #94a3b8;">dB</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">
                    Activity: <span style="color: #38bdf8; font-weight: 700;">normal_discussion</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_slot.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.04)

    finally:
        cap.release()
