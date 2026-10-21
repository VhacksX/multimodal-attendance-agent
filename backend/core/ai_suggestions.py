import os

class AISuggestionEngine:
    """
    EPIC 4 - Activity 4.1: Multimodal Agent Implementation
    Matches the curriculum architecture using LLM completions with fallback reasoning.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                self.client = None

    def _fallback_suggestions(self, session_data: dict) -> dict:
        """
        Deterministic, rule-based fallback reasoning engine when offline.
        """
        score = session_data.get("engagement_score", 0.0)
        sound_state = session_data.get("audio_summary", {}).get("sound_state", "")
        attention = session_data.get("vision_summary", {}).get("attention_state", "")

        if score >= 75.0:
            insight = "Class engagement is high with focused student attention."
            recommendation = "Maintain current pedagogical pace and transition to hands-on exercises."
        elif score >= 50.0:
            if "Distracted" in attention or "Looking Down" in attention:
                insight = "Visual focus is declining, though acoustic environment remains normal."
                recommendation = "Ask a direct checkpoint question or initiate a brief interactive poll."
            elif sound_state == "Active Speech / Discussion":
                insight = "High acoustic activity observed with moderate visual focus."
                recommendation = "Direct class discussion back to the primary lecture slides."
            else:
                insight = "Class status is steady with moderate passive listening."
                recommendation = "Incorporate an illustrative practical example to raise energy."
        else:
            insight = "Student disengagement or prolonged absence detected."
            recommendation = "Pause the lecture, verify attendance, and check for comprehension hurdles."

        return {
            "insight": insight,
            "recommendation": recommendation,
            "confidence": 0.92,
            "source": "Rule-Based Fallback Engine"
        }

    def generate_suggestions(self, session_data: dict) -> dict:
        """
        Generates teaching suggestions from the fused multimodal session packet.
        """
        prompt = (
            f"Classroom Session State:\n"
            f"- Engagement Score: {session_data.get('engagement_score')}/100\n"
            f"- Attention: {session_data.get('vision_summary', {}).get('attention_state')}\n"
            f"- Acoustic State: {session_data.get('audio_summary', {}).get('sound_state')}\n"
            f"- Noise Level: {session_data.get('audio_summary', {}).get('noise_level')}\n"
            f"Provide a 1-sentence diagnostic insight and a 1-sentence actionable recommendation for the teacher."
        )

        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert educational AI teaching assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-8b-8192",
                    temperature=0.7,
                    max_tokens=200
                )
                response = completion.choices[0].message.content.strip()
                return {
                    "insight": response,
                    "recommendation": "Follow AI suggestions based on live metrics.",
                    "confidence": 0.95,
                    "source": "Groq LLM (llama3-8b-8192)"
                }
            except Exception as e:
                print(f"[!] AI Suggestion Error: {e}")
                return self._fallback_suggestions(session_data)
        else:
            return self._fallback_suggestions(session_data)


class InsightGenerator(AISuggestionEngine):
    """Alias to support both naming conventions in the curriculum."""
    def generate_classroom_insight(self, session_data: dict) -> dict:
        return self.generate_suggestions(session_data)
