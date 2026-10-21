import cv2
import os

class FaceDetector:
    """Robust face and attention detector using local Haar cascades."""
    def __init__(self):
        core_dir = os.path.dirname(os.path.abspath(__file__))
        face_path = os.path.join(core_dir, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(core_dir, "haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

    def detect_faces(self, frame):
        """Detects faces without throwing unhandled OpenCV assertion errors."""
        if self.face_cascade is None or self.face_cascade.empty():
            return []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return self.face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
        except Exception:
            return []

    def detect(self, frame):
        return self.detect_faces(frame)

    def classify_attention(self, roi_gray):
        if self.eye_cascade is None or self.eye_cascade.empty():
            return "Attentive"
        try:
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.15, 4, minSize=(16, 16))
            if len(eyes) >= 2:
                return "Attentive"
            elif len(eyes) == 1:
                return "Attentive (Angle)"
            else:
                return "Looking Down / Distracted"
        except Exception:
            return "Attentive"
