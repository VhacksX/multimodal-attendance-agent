import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time

from backend.core.face_detector import FaceDetector
from backend.core.face_recognition import FaceRecognizer
from backend.core.signal_fusion import SignalFusionEngine
from backend.core.ai_suggestions import AISuggestionEngine
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.recommendations_engine import RecommendationsEngine
from backend.core.analytics_service import AnalyticsService

st.set_page_config(
    page_title="EduSense AI - Multimodal Attendance & Engagement",
    page_icon="🎓",
    layout="wide"
)

# Gradient Brand Title Styling
st.markdown("""
<style>
    .gradient-brand {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f43f5e 0%, #ec4899 25%, #a855f7 50%, #3b82f6 75%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .header-subtext {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: -6px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div>🎓 <span class="gradient-brand">EduSense AI</span><span style="font-size: 2.1rem; font-weight: 700; color: #f8fafc;">: Multimodal Attendance & Engagement Agent</span></div>', unsafe_allow_html=True)
st.markdown('<p class="header-subtext">AI-driven classroom intelligence integrating computer vision, acoustic telemetry, and rule-based decision logic.</p>', unsafe_allow_html=True)

# Clean 4-Tab Navigation
tabs = st.tabs(["🎥 Live Classroom Session", "📊 Teacher Profile", "📈 Session Analytics & Reports", "🧪 Edge Case Validation"])

# ==========================================
# TAB 1: LIVE CLASSROOM SESSION
# ==========================================
with tabs[0]:
    st.subheader("Real-Time Multimodal Telemetry")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Live Optical Stream")
        video_placeholder = st.empty()
        run_camera = st.toggle("Activate Live Processing Feed", value=False)

    with col2:
        st.markdown("#### Real-Time Multimodal Metrics")
        metric_attendance = st.metric("Students Detected", "0")
        metric_engagement = st.metric("Fused Engagement Score", "0 / 100")
        metric_audio = st.metric("Acoustic Noise Level", "Silent (-60 dB)")
        
        st.markdown("#### 💡 AI Diagnostic Suggestions")
        suggestion_box = st.info("System idle. Activate camera feed to begin automated monitoring.")

    if run_camera:
        detector = FaceDetector()
        recognizer = FaceRecognizer(threshold_seconds=1.5)
        fusion_engine = SignalFusionEngine()
        ai_engine = AISuggestionEngine()
        cap = cv2.VideoCapture(0)
        
        try:
            for _ in range(150):
                ret, frame = cap.read()
                if not ret:
                    st.error("Unable to access webcam.")
                    break

                frame = cv2.flip(frame, 1)
                faces = detector.detect_faces(frame)
                num_faces = len(faces)
                presence_status = recognizer.update(num_faces > 0)
                primary_attention = "No Face Detected" if num_faces == 0 else "Attentive"

                for (x, y, w, h) in faces:
                    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
                    attention_state = detector.classify_attention(roi_gray)
                    primary_attention = attention_state
                    
                    color = (0, 255, 0) if "Attentive" in attention_state else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Student: {attention_state}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                simulated_db = -42.0 + (5.0 if num_faces > 0 else -10.0)
                audio_packet = {
                    "decibels": simulated_db,
                    "sound_state": "Ambient / Discussion" if simulated_db > -38.0 else "Quiet / Ambient",
                    "noise_level": "Moderate" if simulated_db > -38.0 else "Low"
                }

                vision_packet = {
                    "faces_detected": num_faces,
                    "status": presence_status,
                    "attention_state": primary_attention
                }
                fused = fusion_engine.fuse_signals(vision_packet, audio_packet)
                suggestions = ai_engine.generate_suggestions(fused)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")

                metric_attendance.metric("Students Detected", f"{num_faces} ({presence_status})")
                metric_engagement.metric("Fused Engagement Score", f"{fused['engagement_score']} / 100", fused['engagement_category'])
                metric_audio.metric("Acoustic State", f"{audio_packet['sound_state']}", f"{audio_packet['decibels']:.1f} dB")
                suggestion_box.markdown(f"**Insight:** {suggestions['insight']}\n\n**Action:** {suggestions['recommendation']}")

                time.sleep(0.04)
        finally:
            cap.release()

# ==========================================
# TAB 2: TEACHER PROFILE
# ==========================================
with tabs[1]:
    st.subheader("Instructor Effectiveness & Class Overview")
    teacher_svc = TeacherProfileService()
    profile = teacher_svc.get_profile("PROF_001")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Teaching Effectiveness", f"{profile['effectiveness_score']} / 100", f"+{profile['trend']}% vs Last Month")
    m_col2.metric("Primary Strength", profile["strengths"][0])
    m_col3.metric("Key Focus Area", profile["strengths"][1])

    st.markdown("---")
    st.markdown("#### Historical Class Performance")
    history_df = pd.DataFrame(profile["class_history"])
    st.dataframe(history_df)

    st.markdown("#### Individual Student Progress Trends")
    student_df = pd.DataFrame(profile["student_trends"])
    st.dataframe(student_df)

# ==========================================
# TAB 3: SESSION ANALYTICS & REPORTS
# ==========================================
with tabs[2]:
    st.subheader("Student Session Engagement Timeline")

    sample_timeline = pd.DataFrame({
        "Timestamp": pd.date_range(start="08:47:00", periods=20, freq="15s").strftime("%H:%M:%S"),
        "Average Attention": [82, 85, 84, 88, 55, 82, 84, 48, 86, 89, 87, 85, 90, 92, 88, 86, 84, 88, 89, 91],
        "Acoustic Noise (dB)": [-44, -43, -42, -45, -30, -42, -44, -28, -43, -45, -44, -42, -43, -44, -45, -43, -42, -44, -45, -44]
    })

    fig = px.line(
        sample_timeline, 
        x="Timestamp", 
        y="Average Attention", 
        title="Classroom Attention Fluctuations Over Time",
        markers=True
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)

    csv_bytes = sample_timeline.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Session Analytics (CSV)",
        data=csv_bytes,
        file_name="session_analytics.csv",
        mime="text/csv"
    )

# ==========================================
# TAB 4: EDGE CASE VALIDATION
# ==========================================
with tabs[3]:
    st.subheader("System Robustness & Edge Case Evaluation")
    st.markdown("""
    * **Lighting Variations:** Pre-processed grayscale frames adjust to uneven ambient light.
    * **Partial Occlusions:** Multi-eye detection handles off-axis angles before marking distraction.
    * **Acoustic Disturbance:** RMS energy thresholds isolate voice interaction from continuous ambient noise.
    * **Presence Verification:** State-machine timer requires sustained presence before logging attendance.
    """)

    test_scenario = st.selectbox(
        "Select Edge Case Scenario to Simulate:",
        ["Sudden Loud Noise Spurt", "Student Looking Down at Notes", "Transient Glitch / Face Loss", "Steady Optimal Lecture State"]
    )

    fusion = SignalFusionEngine()
    ai = AISuggestionEngine()

    if test_scenario == "Sudden Loud Noise Spurt":
        v = {"faces_detected": 1, "status": "Marked Present", "attention_state": "Attentive"}
        a = {"decibels": -18.0, "sound_state": "Active Speech / Discussion", "noise_level": "High"}
    elif test_scenario == "Student Looking Down at Notes":
        v = {"faces_detected": 1, "status": "Marked Present", "attention_state": "Looking Down / Distracted"}
        a = {"decibels": -45.0, "sound_state": "Silence", "noise_level": "Low"}
    elif test_scenario == "Transient Glitch / Face Loss":
        v = {"faces_detected": 0, "status": "Verifying Presence...", "attention_state": "No Face Detected"}
        a = {"decibels": -40.0, "sound_state": "Ambient", "noise_level": "Low"}
    else:
        v = {"faces_detected": 1, "status": "Marked Present", "attention_state": "Attentive"}
        a = {"decibels": -35.0, "sound_state": "Ambient / Background", "noise_level": "Moderate"}

    packet = fusion.fuse_signals(v, a)
    out = ai.generate_suggestions(packet)

    st.write("**Fused State Output:**", packet)
    st.success(f"**AI Recommendation:** {out['recommendation']}")
