import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# Step 1: Prepare dataset
# ------------------------------------------------------------
# Folder structure example:
# dataset/
#   age_20/
#       file1.wav, file2.wav ...
#   age_30/
#       file3.wav, file4.wav ...
#   age_40/
#       file5.wav, file6.wav ...

DATASET_DIR = "data"
X, y = [], []

for age_group in os.listdir(DATASET_DIR):
    age_path = os.path.join(DATASET_DIR, age_group)
    if not os.path.isdir(age_path):
        continue

    for file in os.listdir(age_path):
        if file.endswith(".wav"):
            file_path = os.path.join(age_path, file)
            signal, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            features = np.mean(mfcc.T, axis=0)
            X.append(features)
            y.append(age_group)

X = np.array(X)
y = np.array(y)

# ------------------------------------------------------------
# Step 2: Train model
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
age_model = RandomForestClassifier(n_estimators=100, random_state=42)
age_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Step 3: Evaluate
# ------------------------------------------------------------
y_pred = age_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------------------------------------
# Step 4: Save model
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(age_model, "models/age_model.pkl")
print("✅ Age model saved to models/age_model.pkl")

