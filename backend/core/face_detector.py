import cv2
import os

class FaceDetector:
    def __init__(self):
        face_xml = "haarcascade_frontalface_default.xml"
        eye_xml = "haarcascade_eye.xml"

        # Load cascade models
        if os.path.exists(face_xml):
            self.face_cascade = cv2.CascadeClassifier(face_xml)
        else:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if os.path.exists(eye_xml):
            self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        else:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_attention(self, gray_frame, face_box):
        fx, fy, fw, fh = face_box
        upper_face = gray_frame[fy:fy + int(fh * 0.55), fx:fx + fw]
        eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=3, minSize=(18, 18))

        if len(eyes) >= 2:
            return "Attentive"
        elif len(eyes) == 1:
            return "Attentive (Angle)"
        else:
            return "Looking Down / Distracted"

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))

        detections = []
        for (x, y, w, h) in faces:
            attention = self.analyze_attention(gray, (x, y, w, h))
            detections.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "attention": attention
            })
        return detections
