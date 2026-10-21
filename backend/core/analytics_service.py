from datetime import datetime
from .database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric

class AnalyticsService:
    """
    EPIC 5 - Activity 5.2: Student Session Analytics Generation
    Aggregates temporal engagement metrics and downsamples timelines.
    """
    def get_session_trends(self, session_id: int):
        db = SessionLocal()
        try:
            metrics = (
                db.query(PersonMetric)
                .filter(PersonMetric.session_id == session_id)
                .order_by(PersonMetric.timestamp.asc())
                .all()
            )

            timeline = {}
            for m in metrics:
                ts_str = m.timestamp.strftime("%H:%M:%S")
                if ts_str not in timeline:
                    timeline[ts_str] = {"att": [], "count": 0}
                timeline[ts_str]["att"].append(m.attention)
                timeline[ts_str]["count"] += 1

            summary = [
                {
                    "time": t,
                    "avg_attention": round(sum(d["att"]) / len(d["att"]), 1),
                    "active_students": d["count"]
                }
                for t, d in timeline.items()
            ]

            return {
                "session_id": session_id,
                "total_datapoints": len(metrics),
                "timeline": summary
            }
        finally:
            db.close()
