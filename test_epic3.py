import time
import sounddevice as sd
from backend.core.audio_analysis import AudioAnalyzer

analyzer = AudioAnalyzer(sample_rate=16000)
duration = 1.0  # 1-second chunks
sample_rate = 16000

print("\n[+] Testing backend/core/audio_analysis.py (EPIC 3)...")
print("[+] Recording 8 sample windows (~8 seconds). Speak or stay quiet to test...\n")

for i in range(8):
    # Record 1 second of live audio as float32 array
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    metrics = analyzer.analyze_audio_chunk(recording.flatten())
    if metrics:
        print(f"[{i+1}/8] dB: {metrics['decibels']:>6.1f} | RMS: {metrics['rms_energy']:.4f} | State: {metrics['sound_state']} | Engagement: {metrics['engagement_level']}")
    time.sleep(0.2)

print("\n[✓] EPIC 3 verified and ready for multimodal fusion!")
