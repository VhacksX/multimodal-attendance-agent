import cv2
import os

class FaceDetector:
    def __init__(self, face_path=None, eye_path=None):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        default_face = os.path.join(root_dir, "haarcascade_frontalface_default.xml")
        default_eye = os.path.join(root_dir, "haarcascade_eye.xml")

        if face_path is None or not os.path.exists(str(face_path)):
            if os.path.exists(default_face):
                face_path = default_face
            elif hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

        if eye_path is None or not os.path.exists(str(eye_path)):
            if os.path.exists(default_eye):
                eye_path = default_eye
            elif hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(face_path) if face_path and hasattr(cv2, 'CascadeClassifier') else None
        self.eye_cascade = cv2.CascadeClassifier(eye_path) if eye_path and hasattr(cv2, 'CascadeClassifier') else None

    def detect_faces(self, frame):
        if self.face_cascade is None or frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return faces

    def classify_attention(self, face_roi_gray):
        if self.eye_cascade is None or face_roi_gray is None:
            return "engaged"
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=4)
        return "Attentive / Engaged" if len(eyes) > 0 else "Distracted"
