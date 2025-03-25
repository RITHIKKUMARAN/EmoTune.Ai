# Mood Detection

A comprehensive multi-modal mood detection system integrating facial analysis, text interpretation, biometric data, and voice tone analysis to provide personalized music and content recommendations.

## Features

### 🎭 Facial Emotion Analysis
- Detects emotions (angry, disgust, fear, happy, sad, surprise, neutral) from webcam or uploaded images using the **Mini-Xception** model.
- Processes images in real-time or from static uploads with **Mediapipe** for face detection.

### 📝 Text-Based Mood Analysis
- Analyzes user-provided text input to determine mood (happy, sad, calm, excited, neutral) using the **Gemini API**.

### 💓 Biometric Mood Analysis
- Interprets mock or manual biometric data (**heart rate, SpO2, motion**) to infer mood.
- Optionally fetches real-time data from a smartwatch API (currently mocked).

### 🎙️ Voice Tone Analysis
- Analyzes uploaded audio files to detect mood (currently simulated).

### 🔀 Mood Combination
- Combines multiple inputs (facial, text, biometric, voice) with weighted scoring to determine the final mood.

### 🎶 Music Generation
- Generates mood-specific audio (5-second clips) using the **MusicGen** model based on detected mood.
- Provides fallback tone generation (sine wave) if **MusicGen** fails.

### 📻 Content Recommendations
- Fetches **mood-based song recommendations** from the **YouTube API** (5 results max).
- Fetches **mood-based track recommendations** from the **Spotify API** (5 results max).
- Provides **static movie recommendations** based on detected mood.
- Filters recommendations by **user-defined genre preferences**.

### 🎨 User Interface
A **Streamlit-based UI** featuring:
- Mood selection or auto-detection options.
- Text input for mood description.
- Image upload and webcam capture for facial analysis.
- Biometric input (manual or smartwatch integration).
- Audio file upload for voice analysis.
- Generated music playback.
- Recommendation links (YouTube, Spotify, movies).
- **Privacy settings, user insights, mood trends, and a chatbot**.

### 🤖 Chatbot
- Provides an **emotional support chatbot** using the **Gemini API** with **persistent chat history**.

### 👤 User Profile & History
- Tracks **listening history, preferred genres, and mood patterns**.
- Displays user insights (**top mood, favorite genre**).

### 🔒 Privacy & Consent
- Displays a **consent popup** for biometric and facial data usage.
- Allows users to **clear biometric, facial, and chat data**.
- **Processes all data locally without storage**.

### 🎮 Gamification
- Tracks **mood streaks and achievements** (e.g., happy streak, mood explorer badge).
- Saves **emotional memory capsules** (mood, timestamp, generated audio).

### 🌍 Community Features
- Allows users to **share generated music** (simulated).

### 🏥 Global Impact
- Displays **location-based mental health resources** using **IP geolocation**.

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mood-detection.git
cd mood-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🤝 Contributing
Feel free to **fork** this repository and submit a **pull request** with improvements or bug fixes!
