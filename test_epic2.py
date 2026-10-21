import cv2
from backend.core.camera_service import CameraService
from backend.core.face_detector import FaceDetector
from backend.core.face_recognition import AttendanceTracker

cam = CameraService()
cam.start()

detector = FaceDetector()
tracker = AttendanceTracker(sustained_seconds_threshold=5)

print("\n[+] Running EPIC 2 modular vision test...")
print("[+] Press 'q' on the preview window to exit.\n")

while True:
    frame = cam.get_frame()
    if frame is None:
        continue

    detections = detector.process_frame(frame)
    has_face = len(detections) > 0
    status = tracker.update_presence(has_face)

    # Determine overall attention state
    attention_state = detections[0]["attention"] if has_face else "No Face Detected"

    # UI Color cues
    status_color = (0, 255, 0) if "Present" in status else (0, 165, 255)
    attention_color = (0, 255, 0) if "Attentive" in attention_state else (0, 120, 255)

    # Top-Left Telemetry Overlay (Line 1, Line 2, Line 3)
    cv2.putText(frame, f"Attendance: {status}", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
    cv2.putText(frame, f"Attention: {attention_state}", (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, attention_color, 2)
    cv2.putText(frame, f"Faces Detected: {len(detections)}", (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

    # Bounding box around detected face
    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
        cv2.putText(frame, "Student 01", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.imshow("Multimodal Agent - Vision Test (EPIC 2)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
print("[✓] Session completed. Attendance record:", tracker.attendance_log)
