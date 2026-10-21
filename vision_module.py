import cv2
import os
import time
import pandas as pd
from datetime import datetime

class VisionModule:
    """
    EPIC 2: Vision Module Development
    - Activity 2.1: Face detection & attention tracking
    - Activity 2.2: Presence-based sustained duration attendance logic
    """
    def __init__(self, sustained_seconds_threshold=5):
        self.sustained_threshold = sustained_seconds_threshold
        self.first_seen_time = None
        self.attendance_record = {}

        face_xml = "haarcascade_frontalface_default.xml"
        eye_xml = "haarcascade_eye.xml"

        if not os.path.exists(face_xml) or not os.path.exists(eye_xml):
            raise FileNotFoundError("Cascade XML files missing from project root folder.")

        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)

    def analyze_attention(self, gray_frame, face_box):
        fx, fy, fw, fh = face_box
        upper_face = gray_frame[fy:fy + int(fh * 0.55), fx:fx + fw]
        eyes = self.eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(18, 18)
        )

        if len(eyes) >= 2:
            return "Attentive"
        elif len(eyes) == 1:
            return "Attentive (Angle)"
        else:
            return "Looking Down / Distracted"

    def process_frame(self, frame):
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()

        telemetry = {
            "faces_detected": 0,
            "status": "Absent",
            "attention_state": "No Face Detected",
            "duration_seconds": 0.0
        }

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(70, 70)
        )

        if len(faces) > 0:
            telemetry["faces_detected"] = len(faces)
            primary_face = faces[0]
            fx, fy, fw, fh = primary_face

            if self.first_seen_time is None:
                self.first_seen_time = current_time

            duration = current_time - self.first_seen_time
            telemetry["duration_seconds"] = round(duration, 1)

            if duration >= self.sustained_threshold:
                telemetry["status"] = "Marked Present"
                if "Student_01" not in self.attendance_record:
                    self.attendance_record["Student_01"] = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Status": "Present",
                        "Duration_Sec": round(duration, 1)
                    }
            else:
                telemetry["status"] = f"Verifying ({int(duration)}s/{self.sustained_threshold}s)"

            telemetry["attention_state"] = self.analyze_attention(gray, primary_face)

            box_color = (0, 255, 0) if "Present" in telemetry["status"] else (0, 165, 255)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
            cv2.putText(frame, "Student 01", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        else:
            self.first_seen_time = None
            telemetry["status"] = "Marked Present" if "Student_01" in self.attendance_record else "Absent"

        return telemetry

    def export_attendance_csv(self, filename="attendance_log.csv"):
        if self.attendance_record:
            df = pd.DataFrame.from_dict(self.attendance_record, orient="index")
            df.to_csv(filename, index_label="Student_ID")
            print(f"\n[✓] Attendance exported successfully to {filename}")
        else:
            print("\n[!] No records logged.")
        return filename


if __name__ == "__main__":
    print("\n[+] Initializing Vision Module...")
    tracker = VisionModule(sustained_seconds_threshold=5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Unable to access camera.")
        exit()

    print("[+] Camera running with horizontal mirror enabled.")
    print("[+] Press 'q' on the preview window to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab video frame.")
            break

        # Mirror video horizontally for a natural webcam view
        frame = cv2.flip(frame, 1)

        data = tracker.process_frame(frame)

        status_color = (0, 255, 0) if "Present" in data["status"] else (0, 165, 255)
        attention_color = (0, 255, 0) if "Attentive" in data["attention_state"] else (0, 120, 255)

        cv2.putText(frame, f"Attendance: {data['status']}", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cv2.putText(frame, f"Attention: {data['attention_state']}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, attention_color, 2)
        cv2.putText(frame, f"Faces Detected: {data['faces_detected']}", (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

        cv2.imshow("Multimodal Agent - Vision Test (EPIC 2)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.export_attendance_csv()
