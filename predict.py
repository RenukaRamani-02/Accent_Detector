import os
import joblib
import librosa
import numpy as np

# ------------------------------------------------------------
# Safe path loading for models
# ------------------------------------------------------------
base_dir = os.path.dirname(__file__)

accent_model_path = os.path.join(base_dir, "models", "mfcc_model.pkl")
age_model_path = os.path.join(base_dir, "models", "age_model.pkl")

# Load models if they exist
accent_model = joblib.load(accent_model_path) if os.path.exists(accent_model_path) else None
age_model = joblib.load(age_model_path) if os.path.exists(age_model_path) else None

# ------------------------------------------------------------
# Prediction functions
# ------------------------------------------------------------
def predict_accent(audio_file):
    if accent_model is None:
        return "Accent model not available", 0.0
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = accent_model.predict(features)[0]

    # Confidence score if model supports probabilities
    if hasattr(accent_model, "predict_proba"):
        proba = accent_model.predict_proba(features)
        confidence = float(np.max(proba[0]))
    else:
        confidence = 1.0

    return prediction, confidence


def predict_age(audio_file):
    if age_model is None:
        return "Age model not available", 0.0
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = age_model.predict(features)[0]

    # Confidence score if model supports probabilities
    if hasattr(age_model, "predict_proba"):
        proba = age_model.predict_proba(features)
        confidence = float(np.max(proba[0]))
    else:
        confidence = 1.0

    return prediction, confidence


