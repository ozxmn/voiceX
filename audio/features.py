# audio/features.py
import numpy as np
import librosa
from pathlib import Path

SAMPLE_RATE = 16000
DEFAULT_RECORD_SECONDS = 3
N_MFCC = 13
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def extract_features(wav_path: Path):
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio file")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)
