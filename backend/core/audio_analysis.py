import numpy as np
import os

class AudioAnalyzer:
    """
    EPIC 3: Audio Module Development
    - Activity 3.1: Audio Signal Extraction & Normalization
    - Activity 3.2: Noise & Speech Activity Analysis
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def analyze_audio_chunk(self, audio_data):
        """
        Analyzes raw audio bytes/array:
        1. Noise level (decibels)
        2. Speech vs silence ratio
        3. Sound activity patterns
        
        Returns engagement metrics
        """
        try:
            # Convert raw 16-bit PCM bytes to float32 numpy array if needed
            if isinstance(audio_data, bytes):
                y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                y = np.array(audio_data, dtype=np.float32)

            if len(y) == 0:
                return None

            # Calculate RMS energy and convert to decibels (dBFS)
            rms = float(np.sqrt(np.mean(np.square(y))))
            db = float(20 * np.log10(rms + 1e-6))

            # Classify acoustic states based on decibel energy levels
            if db < -45.0:
                sound_state = "Silence"
                noise_level = "Low"
                engagement = "Passive"
            elif db < -25.0:
                sound_state = "Ambient / Background"
                noise_level = "Moderate"
                engagement = "Attentive"
            else:
                sound_state = "Active Speech / Discussion"
                noise_level = "High"
                engagement = "Active"

            return {
                "rms_energy": round(rms, 4),
                "decibels": round(db, 2),
                "sound_state": sound_state,
                "noise_level": noise_level,
                "engagement_level": engagement,
                "samples_processed": len(y)
            }
        except Exception as e:
            print(f"Error analyzing audio chunk: {e}")
            return None
