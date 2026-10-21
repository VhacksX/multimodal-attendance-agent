from backend.core.signal_fusion import SignalFusionEngine
from backend.core.ai_suggestions import AISuggestionEngine

fusion = SignalFusionEngine()
agent = AISuggestionEngine()

print("\n=======================================================")
print("     MULTIMODAL AGENT & SIGNAL FUSION TEST (EPIC 4)    ")
print("=======================================================\n")

# Simulated test scenario: Attentive student during lecture
sample_vision = {
    "faces_detected": 1,
    "status": "Marked Present",
    "attention_state": "Attentive"
}

sample_audio = {
    "decibels": -34.5,
    "sound_state": "Ambient / Background",
    "noise_level": "Moderate"
}

fused_packet = fusion.fuse_signals(sample_vision, sample_audio)
ai_output = agent.generate_suggestions(fused_packet)

print(f"[Timestamp]           : {fused_packet['timestamp']}")
print(f"[Engagement Score]    : {fused_packet['engagement_score']} / 100 ({fused_packet['engagement_category']})")
print(f"[Visual Modality]     : {fused_packet['vision_summary']['attendance_status']} | {fused_packet['vision_summary']['attention_state']}")
print(f"[Acoustic Modality]   : {fused_packet['audio_summary']['sound_state']} ({fused_packet['audio_summary']['decibels']} dB)")
print(f"\n[AI Diagnostic Insight]:")
print(f"  → {ai_output['insight']}")
print(f"[AI Recommendation]   :")
print(f"  → {ai_output['recommendation']}")
print(f"[Engine Source]       : {ai_output['source']}\n")
print("[✓] EPIC 4 verification complete!")
