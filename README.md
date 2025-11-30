🎙️ Accent, Age & Cuisine Detection App
📖 Overview
This project is a Streamlit-based machine learning app that detects a speaker’s regional accent, predicts their age group, and maps them to a cuisine recommendation from their region. It combines speech feature extraction, ML classification models, and a cultural cuisine map to deliver an engaging, user-friendly experience.

🚀 Features
✨ Accent detection for multiple Indian regions (Tamil, Kerala, Karnataka, Jharkhand, Gujarat, Andhra, etc.) ✨ Age prediction from voice samples ✨ Cuisine map integration linked to detected accent ✨ Balanced dataset handling to reduce bias ✨ Streamlit interface for easy interaction

🛠️ Installation
1️⃣ Clone the repository → git clone https://github.com/RenukaRamani-02/Accent_Detector

2️⃣ Create & activate virtual environment
We use accentvenv as the environment name.

Windows (Command Prompt):
bash
python -m venv accentvenv
accentvenv\Scripts\activate

Windows (PowerShell):
bash
python -m venv accentvenv
.\accentvenv\Scripts\activate

macOS/Linux:
bash
python3 -m venv accentvenv
source accentvenv/bin/activate
 
 👉 You’ll see (accentvenv) at the start of your terminal prompt when activated.

▶️ Usage
▶️ Run the app locally → streamlit run app.py 🌐 Open your browser at http://localhost:8501 to interact with the app.

📂 Folder Structure
📁 project/

📄 app.py → Main Streamlit app

📁 models/ → ML models for accent & age detection

📁 data/ → Training and testing datasets

📁 scripts/ → Helper scripts for automation & retraining

📄 cuisine_map.json → Accent-to-cuisine mapping

📄 requirements.txt → Python dependencies

📄 README.md → Project documentation


Accent_Detector/ ├── app.py # Streamlit app ├── predict.py # Prediction functions ├── train_models.py # Combined training script ├── test_predict.py # Local test script ├── requirements.txt # Dependencies ├── models/ # Saved models (.pkl) │ ├── mfcc_model.pkl │ └── age_model.pkl └── data/ # Training data (folders of .wav files)


⚙️ Dependencies
🐍 Python 3.9+ 📊 Streamlit 🧠 Scikit-learn 📑 Pandas, NumPy 🎵 Librosa (for audio feature extraction) 📈 Matplotlib / Seaborn (for visualization)

📊 Workflow
🔹 Data Preprocessing → Audio samples → MFCC feature extraction → Balanced dataset automation 🔹 Model Training → Accent classification model + Age prediction model 🔹 Cuisine Mapping → Accent → Region → Cuisine recommendation 🔹 Streamlit Integration → Upload audio → Get predictions → Display cuisine map

🌱 Future Improvements
🌍 Expand accent coverage to more regions 🍲 Enhance cuisine recommendations with cultural context ☁️ Deploy as a cloud-hosted app for wider accessibility 🗣️ Add multilingual support

🤝 Contributing
💡 Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License
📄 This project is licensed under the MIT License.
