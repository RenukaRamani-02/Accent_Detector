
import os
from predict import predict_accent, predict_age

# Find the first available .wav file in the data/ folder
def find_first_wav(root="data"):
    for r, d, f in os.walk(root):
        for file in f:
            if file.lower().endswith(".wav"):
                return os.path.join(r, file)
    return None

test_audio = find_first_wav()

if test_audio is None:
    print("❌ No .wav files found in the data/ folder.")
else:
    print(f"🔎 Testing with: {test_audio}")
    accent_label, accent_conf = predict_accent(test_audio)
    print(f"Accent: {accent_label}, Confidence: {accent_conf:.2f}%")
    age_label, age_conf = predict_age(test_audio)
    print(f"Age Group: {age_label}, Confidence: {age_conf:.2f}%")
