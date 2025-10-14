# model/recognizer.py
import pickle
from pathlib import Path
from audio.features import extract_features
from model.trainer import MODEL_PATH

def recognize_file(wav_path: Path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet.")
    with open(MODEL_PATH, "rb") as f:
        model, scaler = pickle.load(f)
    feat = extract_features(wav_path).reshape(1, -1)
    feat = scaler.transform(feat)
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0].max()
    return pred, prob
