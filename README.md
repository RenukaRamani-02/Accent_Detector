🎙️ Accent, Age & Cuisine Detection App
📖 Overview
This project is a Streamlit-based machine learning app that detects a speaker’s regional accent, predicts their age group, and maps them to a cuisine recommendation from their region. It combines speech feature extraction, ML classification models, and a cultural cuisine map to deliver an engaging, user-friendly experience.

🚀 Features
✨ Accent detection for multiple Indian regions (Tamil, Kerala, Karnataka, Jharkhand, Gujarat, Andhra, etc.) ✨ Age prediction from voice samples ✨ Cuisine map integration linked to detected accent ✨ Balanced dataset handling to reduce bias ✨ Streamlit interface for easy interaction

🛠️ Installation
1️⃣ Clone the repository → git clone https://github.com/https://github.com/RenukaRamani-02/Accent_Detector/accent-age-cuisine-app.git 2️⃣ Create a virtual environment → python -m venv venv

Activate: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows) 3️⃣ Install dependencies → pip install -r requirements.txt

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
