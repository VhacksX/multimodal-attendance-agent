import time
from datetime import datetime

class SignalFusionEngine:
    """
    EPIC 4 - Activity 4.2: Multimodal Signal Fusion
    EPIC 4 - Activity 4.3: Engagement Scoring Logic
    """
    def __init__(self):
        self.session_history = []

    def compute_engagement_score(self, vision_data: dict, audio_data: dict) -> float:
        """
        Activity 4.3: Engagement Scoring Logic
        Weighted calculation:
        - Presence & Continuity: 40 points
        - Gaze & Visual Attention: 35 points
        - Classroom Acoustic Dynamics: 25 points
        """
        score = 0.0

        # 1. Visual Presence Component (Max 40 pts)
        if vision_data.get("faces_detected", 0) > 0:
            status = vision_data.get("status", "")
            if "Present" in status:
                score += 40.0
            else:
                score += 20.0  # In verification phase

        # 2. Visual Attention Component (Max 35 pts)
        attention = vision_data.get("attention_state", "")
        if "Attentive" in attention and "Angle" not in attention:
            score += 35.0
        elif "Angle" in attention:
            score += 25.0
        elif "Looking Down" in attention or "Distracted" in attention:
            score += 10.0

        # 3. Acoustic Environment Component (Max 25 pts)
        sound_state = audio_data.get("sound_state", "")
        noise_level = audio_data.get("noise_level", "")

        if sound_state == "Active Speech / Discussion":
            score += 25.0  # High interactive engagement
        elif sound_state == "Ambient / Background" or noise_level == "Low":
            score += 20.0  # Quiet, focused listening environment
        elif noise_level == "High":
            score += 5.0   # Excessive ambient disturbance

        return round(min(100.0, score), 1)

    def fuse_signals(self, vision_data: dict, audio_data: dict) -> dict:
        """
        Activity 4.2: Fuses visual and acoustic telemetry with a shared timestamp.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        engagement_score = self.compute_engagement_score(vision_data, audio_data)

        # Categorize overall engagement band
        if engagement_score >= 75.0:
            category = "High Engagement"
        elif engagement_score >= 50.0:
            category = "Moderate Engagement"
        else:
            category = "Low / Disengaged"

        fused_packet = {
            "timestamp": timestamp,
            "engagement_score": engagement_score,
            "engagement_category": category,
            "vision_summary": {
                "faces_detected": vision_data.get("faces_detected", 0),
                "attendance_status": vision_data.get("status", "Absent"),
                "attention_state": vision_data.get("attention_state", "No Face Detected")
            },
            "audio_summary": {
                "decibels": audio_data.get("decibels", -60.0),
                "sound_state": audio_data.get("sound_state", "Silence"),
                "noise_level": audio_data.get("noise_level", "Low")
            }
        }

        self.session_history.append(fused_packet)
        return fused_packet
