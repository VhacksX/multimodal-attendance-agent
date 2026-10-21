class RecommendationsEngine:
    """
    EPIC 5 - Activity 5.1: Real-time Rule-Based Recommendations
    Evaluates participant attention levels and outputs diagnostic alerts.
    """
    def generate_realtime_recommendations(self, session_status: dict):
        recs = []
        people = session_status.get('people', [])

        if not people:
            return [{'type': 'info', 'message': 'Waiting for participant detection...'}]

        # 1. Individual attention check
        avg_att = sum(p.get('attention', 0) for p in people) / len(people)

        for p in people:
            att = p.get('attention', 0)
            sid = p.get('id', 'Unknown')
            if att < 40:
                recs.append({
                    'type': 'alert',
                    'message': f"Student {sid} is distracted ({att}%)."
                })

        # 2. Global class attention check
        if len(people) >= 1 and avg_att < 60:
            recs.append({
                'type': 'warning',
                'message': f"Class attention is low ({int(avg_att)}%). Consider an interaction break."
            })
        elif avg_att >= 80:
            recs.append({
                'type': 'info',
                'message': f"High classroom focus ({int(avg_att)}%). Ideal pace to continue."
            })

        return recs
