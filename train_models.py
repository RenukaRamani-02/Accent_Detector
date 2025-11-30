import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# ------------------------------------------------------------
# Accent model training
# ------------------------------------------------------------
accent_dir = "data_accent"
X_accent, y_accent = [], []

for region in os.listdir(accent_dir):
    region_path = os.path.join(accent_dir, region)
    if os.path.isdir(region_path):
        for file in os.listdir(region_path):
            if file.endswith(".wav"):
                features = extract_features(os.path.join(region_path, file))
                X_accent.append(features)
                y_accent.append(region)

X_train, X_test, y_train, y_test = train_test_split(X_accent, y_accent, test_size=0.2, random_state=42)
accent_model = RandomForestClassifier(n_estimators=100, random_state=42)
accent_model.fit(X_train, y_train)
joblib.dump(accent_model, os.path.join(models_dir, "mfcc_model.pkl"))
print("✅ Accent model trained. Accuracy:", accent_model.score(X_test, y_test))

# ------------------------------------------------------------
# Age model training
# ------------------------------------------------------------
age_dir = "data_age"
X_age, y_age = [], []

for age_group in os.listdir(age_dir):
    age_path = os.path.join(age_dir, age_group)
    if os.path.isdir(age_path):
        for file in os.listdir(age_path):
            if file.endswith(".wav"):
                features = extract_features(os.path.join(age_path, file))
                X_age.append(features)
                y_age.append(age_group)

X_train, X_test, y_train, y_test = train_test_split(X_age, y_age, test_size=0.2, random_state=42)
age_model = RandomForestClassifier(n_estimators=100, random_state=42)
age_model.fit(X_train, y_train)
joblib.dump(age_model, os.path.join(models_dir, "age_model.pkl"))
print("✅ Age model trained. Accuracy:", age_model.score(X_test, y_test))
