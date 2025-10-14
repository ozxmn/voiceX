# model/trainer.py
import pickle
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

from audio.features import extract_features, DATA_DIR

MODEL_PATH = Path("model.pkl")

def train_model_from_folder(status_callback=None):
    X, y = [], []
    for folder in DATA_DIR.iterdir():
        if not folder.is_dir():
            continue
        label = folder.name
        for wav in folder.glob("*.wav"):
            try:
                feat = extract_features(wav)
                if feat is None or np.isnan(feat).any():
                    continue
                X.append(feat)
                y.append(label)
                if status_callback:
                    status_callback(f"Loaded {wav.name} ({label})")
            except Exception as e:
                if status_callback:
                    status_callback(f"Error reading {wav.name}: {e}")

    if len(set(y)) < 2:
        raise ValueError("Need at least two distinct speakers.")

    for lbl, count in Counter(y).items():
        if count < 3:
            raise ValueError(f"Not enough samples for {lbl} (need ≥3)")

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, stratify=y)
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler), f)
    return acc
