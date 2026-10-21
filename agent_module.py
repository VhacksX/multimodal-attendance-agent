"""
EduSense AI - Multimodal Classroom Agent Engine
Coordinates visual attention metrics, acoustic telemetry, and dynamic classroom interventions.
"""

import time
import numpy as np

class MultimodalAttendanceAgent:
    def __init__(self, engagement_decay=0.95):
        self.engagement_decay = engagement_decay
        self.session_start = time.time()
        self.student_scores = {}
        self.history = []

    def evaluate_classroom_state(self, num_faces, engaged_count, noise_level_db):
        """
        Fuses visual telemetry (facial presence, attention status) with acoustic context.
        Returns aggregate engagement score, status verdict, and prioritized teaching recommendation.
        """
        if num_faces == 0:
            return {
                "status": "standby",
                "engagement_score": 0.0,
                "banner_badge": "STANDBY",
                "message": "Waiting for students to join optical frame...",
                "priority": "LOW"
            }

        # Calculate attention ratio
        attention_ratio = engaged_count / max(1, num_faces)
        base_score = attention_ratio * 100.0

        # Acoustic context evaluation
        is_noisy = noise_level_db > -35.0
        
        if attention_ratio >= 0.70 and not is_noisy:
            return {
                "status": "optimal",
                "engagement_score": round(base_score, 1),
                "banner_badge": "SUCCESS",
                "message": "Engagement is optimal. Keep going!",
                "suggestion": "Challenge students with harder problem",
                "impact": "Deep Learning",
                "priority": "HIGH"
            }
        elif is_noisy and attention_ratio < 0.50:
            return {
                "status": "distracted",
                "engagement_score": round(base_score, 1),
                "banner_badge": "ALERT",
                "message": "High acoustic disturbance and low visual attention detected.",
                "suggestion": "Initiate quiet reflection or focused problem-solving break",
                "impact": "Re-engagement",
                "priority": "CRITICAL"
            }
        else:
            return {
                "status": "moderate",
                "engagement_score": round(base_score, 1),
                "banner_badge": "ATTENTION",
                "message": "Student attention is drifting. Suggest quick interaction break.",
                "suggestion": "Conduct interactive poll or question check-in",
                "impact": "Classroom Dynamics",
                "priority": "MEDIUM"
            }

    def update_leaderboard(self, student_name, is_attentive):
        """Increments score for active, attentive students."""
        if student_name not in self.student_scores:
            self.student_scores[student_name] = 0
            
        if is_attentive:
            self.student_scores[student_name] += 1
            
        return sorted(self.student_scores.items(), key=lambda x: x[1], reverse=True)
