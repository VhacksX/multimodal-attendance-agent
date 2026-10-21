import time

class FaceRecognizer:
    """
    EPIC 2 / EPIC 6: Sustained Presence & Attendance Verification
    Tracks face detections over a temporal threshold before marking 'Present'.
    """
    def __init__(self, threshold_seconds: float = 1.5):
        self.threshold_seconds = threshold_seconds
        self.first_detected_time = None
        self.is_present = False

    def update(self, face_detected: bool) -> str:
        """
        Updates verification state based on real-time face detection stream.
        Returns status string: 'Marked Present', 'Verifying Presence...', or 'Absent'
        """
        now = time.time()

        if face_detected:
            if self.first_detected_time is None:
                self.first_detected_time = now

            elapsed = now - self.first_detected_time
            if elapsed >= self.threshold_seconds:
                self.is_present = True
                return "Marked Present"
            else:
                return f"Verifying Presence ({elapsed:.1f}s)..."
        else:
            self.first_detected_time = None
            self.is_present = False
            return "Absent"

    def reset(self):
        self.first_detected_time = None
        self.is_present = False

# Backward compatibility alias
AttendanceTracker = FaceRecognizer
