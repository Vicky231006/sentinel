# utils.py - Helper functions (optional)

import numpy as np
import librosa

def validate_audio_input(data):
    """Validate incoming audio data"""
    if not data or 'audio' not in data:
        return False, "No audio data provided"
    return True, "Valid"

def get_audio_info(audio_data, sample_rate):
    """Get basic audio information"""
    duration = len(audio_data) / sample_rate
    return {
        'duration_seconds': round(duration, 2),
        'sample_rate': sample_rate,
        'total_samples': len(audio_data)
    }