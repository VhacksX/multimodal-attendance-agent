import os
import time
from datetime import datetime

class AttendanceTracker:
    def __init__(self, sustained_seconds_threshold=5):
        self.sustained_threshold = sustained_seconds_threshold
        self.presence_timer = None
        self.attendance_log = {}

    def update_presence(self, face_detected, student_id="Student_01"):
        now = time.time()
        status = "Absent"

        if face_detected:
            if self.presence_timer is None:
                self.presence_timer = now

            duration = now - self.presence_timer
            if duration >= self.sustained_threshold:
                status = "Marked Present"
                if student_id not in self.attendance_log:
                    self.attendance_log[student_id] = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Status": "Present"
                    }
            else:
                status = f"Verifying ({int(duration)}s/{self.sustained_threshold}s)"
        else:
            self.presence_timer = None
            if student_id in self.attendance_log:
                status = "Marked Present"

        return status
