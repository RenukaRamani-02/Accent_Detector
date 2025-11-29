import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Path to balanced dataset
DATASET_DIR = "data"
MODEL_PATH = "models/mfcc_model.pkl"

def extract_features(file_path):
    # Load audio with consistent sample rate
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def load_dataset(dataset_dir):
    X, y = [], []
    for accent in os.listdir(dataset_dir):
        accent_path = os.path.join(dataset_dir, accent)
        if os.path.isdir(accent_path):
            for file in os.listdir(accent_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(accent_path, file)
                    try:
                        features = extract_features(file_path)
                        X.append(features)
                        y.append(accent)
                    except Exception as e:
                        print(f"⚠️ Error processing {file_path}: {e}")
    return np.array(X), np.array(y)

def train_model():
    print("📂 Loading dataset...")
    X, y = load_dataset(DATASET_DIR)

    print("🔀 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🌲 Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    print("✅ Saving model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    train_model()

