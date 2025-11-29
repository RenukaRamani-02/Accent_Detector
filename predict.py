import joblib
import librosa
import numpy as np

# Load your trained accent model
accent_model = joblib.load("models/mfcc_model.pkl")

def extract_features(file_path: str):
    """Extract MFCC features from audio file."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

def predict_accent(file_path: str):
    """Predict accent and confidence using trained accent model."""
    features = extract_features(file_path)
    accent_label = accent_model.predict(features)[0]
    confidence = np.max(accent_model.predict_proba(features)) * 100
    return accent_label, confidence

def predict_age(file_path: str):
    """Dummy age predictor — no model file needed."""
    import random
    age_groups = ["Young (18-30)", "Adult (31-50)", "Senior (51+)"]
    return random.choice(age_groups)

