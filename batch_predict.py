import os
from predict import predict_accent

# 👇 List all accent folders you want to test
folders = ["data/gujrat", "data/kerala", "data/tamil", "data/andhra_pradesh", "data/karnataka", "data/jharkhand"]

# Open a results file for writing (no emoji, plain text)
with open("results.txt", "w", encoding="utf-8") as report:
    # Loop through each folder
    for folder in folders:
        print(f"\nChecking folder: {folder}")
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            report.write(f"{folder}: Folder not found\n")
            continue

        total = 0
        correct = 0
        accent_name = os.path.basename(folder)

        # Loop through all .wav files in the folder
        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder, filename)
                try:
                    predicted = predict_accent(file_path)
                    total += 1
                    if predicted.lower() == accent_name.lower():
                        correct += 1
                    print(f"[{accent_name}] {filename} → {predicted}")
                except Exception as e:
                    print(f"[{accent_name}] {filename} → Error: {e}")

        # Print and save summary for this folder
        if total > 0:
            accuracy = (correct / total) * 100
            summary = f"{accent_name}: {correct}/{total} correct ({accuracy:.2f}%)"
            print(summary)
            report.write(summary + "\n")
