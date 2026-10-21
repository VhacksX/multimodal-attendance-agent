from datetime import datetime
from backend.core.database import SessionLocal, PersonMetric
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.recommendations_engine import RecommendationsEngine
from backend.core.analytics_service import AnalyticsService

print("\n=======================================================")
print("      EPIC 5: ANALYTICS & RECOMMENDATION TEST         ")
print("=======================================================")

# Activity 5.1: Teacher Profile Service
teacher_svc = TeacherProfileService()
profile = teacher_svc.get_profile("PROF_001")
print(f"\n[Teacher Profile] ID: {profile['teacher_id']} | Effectiveness: {profile['effectiveness_score']}%")
print(f"  → Strengths: {', '.join(profile['strengths'])}")
print(f"  → Class: {profile['class_history'][0]['name']} (Score: {profile['class_history'][0]['engagement']}%)")

# Activity 5.1: Real-time Recommendations
rec_engine = RecommendationsEngine()
sample_status = {
    "people": [
        {"id": "Student_01", "attention": 85},
        {"id": "Student_02", "attention": 32}
    ]
}
recommendations = rec_engine.generate_realtime_recommendations(sample_status)
print("\n[Real-Time Recommendations]:")
for r in recommendations:
    print(f"  [{r['type'].upper()}] {r['message']}")

# Activity 5.2: Student Session Analytics Generation
db = SessionLocal()
db.query(PersonMetric).delete()
db.add_all([
    PersonMetric(session_id=1, student_id="Student_01", attention=85.0, timestamp=datetime.utcnow()),
    PersonMetric(session_id=1, student_id="Student_02", attention=32.0, timestamp=datetime.utcnow())
])
db.commit()
db.close()

analytics_svc = AnalyticsService()
trends = analytics_svc.get_session_trends(1)
print("\n[Student Session Trends]:")
print(f"  → Session ID: {trends['session_id']}")
print(f"  → Total Data Points: {trends['total_datapoints']}")
for point in trends["timeline"]:
    print(f"  → Time: {point['time']} | Average Attention: {point['avg_attention']}%")

print("\n[✓] EPIC 5 verification complete!\n")
