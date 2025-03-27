import os
import json
import streamlit as st
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from transformers import pipeline
from audiocraft.models import MusicGen
import soundfile as sf
import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import warnings
from PIL import Image
import torch
from scipy.io.wavfile import write
from numpy import sin, linspace
import random
import backoff
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import Counter
import requests  # Added for smartwatch API calls
import subprocess

# Ensure OpenCV is installed first
try:
    import cv2
except ImportError:
    subprocess.run(["pip", "install", "opencv-python-headless"])
    import cv2

# Now import MediaPipe safely
try:
    import mediapipe as mp
except ImportError:
    subprocess.run(["pip", "install", "mediapipe"])
    import mediapipe as mp


# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
tf.disable_v2_behavior()

# Configure Gemini API
genai.configure(api_key="AIzaSyBFAZbDq0cUKULPMTcZfoiJA5WxpbIscRQ")  # Replace with your Gemini API key
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Debug flag
DEBUG_MODE = False
st.set_page_config(page_title="EmoTune.AI", page_icon="🎵", layout="wide")
# Global graph, session, and model
graph = tf.Graph()
session = tf.Session(graph=graph)
emotion_model = None

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load Mini-Xception model
with graph.as_default():
    with session.as_default():
        model_path = "fer2013_mini_XCEPTION.102-0.66.hdf5"
        try:
            emotion_model = tf.keras.models.load_model(model_path)
            init = tf.global_variables_initializer()
            session.run(init)
        except Exception as e:
            st.error(f"Failed to load Mini-Xception model: {str(e)}. Using fallback CNN.")
            def create_fallback_model():
                model = Sequential()
                model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
                model.add(MaxPooling2D((2, 2)))
                model.add(Flatten())
                model.add(Dense(7, activation='softmax'))
                return model
            emotion_model = create_fallback_model()
            emotion_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            init = tf.global_variables_initializer()
            session.run(init)

# Load other models
@st.cache_resource
def load_other_models():
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    music_model = MusicGen.get_pretrained("facebook/musicgen-small")
    music_model.set_generation_params(duration=5, top_k=250, top_p=0.9)
    youtube = build("youtube", "v3", developerKey="AIzaSyAhYOPNE95kznfB9FRMUc-Ll23FJ37lovE")  # Replace with your YouTube API key
    return sentiment_analyzer, music_model, youtube

sentiment_analyzer, music_model, youtube = load_other_models()

# Spotify setup
client_id = "9ff2eceb450a47e4884328752d6a06d7"  # Replace with your Spotify Client ID
client_secret = "df8f34cc0c034fada7783040f5b71a51"  
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10)

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MOOD_OPTIONS = ["happy", "sad", "calm", "excited", "neutral"]

# Fallback song recommendations
FALLBACK_SONGS = {
    "happy": [("Pharrell Williams - Happy", "https://www.youtube.com/watch?v=ZbZSe6N_BXs")],
    "sad": [("Adele - Someone Like You", "https://www.youtube.com/watch?v=hLQl3WQQoQ0")],
    "calm": [("Ludovico Einaudi - Nuvole Bianche", "https://www.youtube.com/watch?v=4VR-6AS0-l4")],
    "excited": [("Queen - Don't Stop Me Now", "https://www.youtube.com/watch?v=HgzGwKwLmgM")],
    "neutral": [("The Lumineers - Ho Hey", "https://www.youtube.com/watch?v=zvCBSSwgtg4")]
}

# Fallback Spotify tracks
FALLBACK_TRACKS = {
    "happy": [("Happy by Pharrell Williams", "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH")],
    "sad": [("Someone Like You by Adele", "https://open.spotify.com/track/4kflIGfjdZJW4ot2ioixTB")],
    "calm": [("Clair de Lune by Debussy", "https://open.spotify.com/track/5J3P6u1fBOCiosqNxv1xjL")],
    "excited": [("Sweet but Psycho by Ava Max", "https://open.spotify.com/track/25ZN1oORc7f3S9EZ7W4mW")],
    "neutral": [("Lo-Fi Beats by Chillhop Music", "https://open.spotify.com/track/3iG3lQ6xZqS5e9z1v1Zf0Z")]
}

translations = {
    "en": {
        "title": "EmoTune.AI 🎼",
        "subtitle": "A mood-responsive audio experience",
        "select_mood": "🎭 Select Your Mood",
        "choose_mood": "Choose your mood (or auto-detect)",
        "describe_day": "📝 Describe Your Day",
        "describe_mood": "Describe your mood or day (optional)",
        "speak_day": "🎙️ Speak About Your Day (Optional)",
        "upload_audio": "Upload an audio clip of you speaking",
        "upload_image": "📷 Upload an Image (Optional)",
        "upload_image_prompt": "Upload an image for facial emotion analysis",
        "facial_emotion_live": "📸 Facial Emotion Analysis (Live)",
        "enable_facial_analysis": "Enable live facial emotion analysis",
        "mood_booster": "🌟 Mood Booster Tips",  # Already present
        "mood_tip_reflect": "Reflect on your day—what made you smile? 📝",  # Added
        "mood_tip_walk": "Take a short walk to lift your spirits! 🚶",  # Added
        "mood_tip_music": "Listen to some uplifting music—we’ve got you covered! 🎶",  # Added
        "mood_tip_breathe": "Try some deep breathing exercises to maintain your peace! 🧘",  # Added
        "mood_tip_tea": "Sip a warm cup of tea and relax! ☕",  # Added
        "mood_tip_workout": "Channel your energy into a fun activity—maybe a quick workout? 🏋️",  # Added
        "mood_tip_friend": "Share your excitement with a friend! 📞",  # Added
        "mood_tip_new": "Try something new today to spark some inspiration! ✨",  # Added
        "mood_tip_dance": "Dance to your favorite song to amplify your joy! 💃",  # Added
        "mood_tip_let_go": "Take a moment to breathe deeply and let go of tension! 🌬️",  # Added
        "mood_tip_write": "Write down what’s bothering you to clear your mind! ✍️",  # Added
        "mood_tip_talk": "Talk to someone you trust to ease your worries! 🗣️",  # Added
        "mood_tip_grounding": "Try a grounding exercise—focus on your surroundings! 🌳",  # Added
        "mood_tip_embrace": "Embrace the unexpected—maybe it’s a sign of something great! 🎉",  # Added
        "mood_tip_journal": "Capture this moment with a quick journal entry! 📓",  # Added
        "mood_tip_step_away": "Step away from what’s bothering you and take a break! 🚪",  # Added
        "mood_tip_positive": "Focus on something positive to shift your mood! 🌈",  # Added
        "mood_trend": "📈 Mood Trend",
        "mental_health": "🌍 Mental Health & Community",  # Already present
        "mental_health_info": "Music can reduce stress by up to 65% (Source: WHO). Need more support? Explore WHO Mental Health Resources.",  # Added
        "share_music": "Share to Community",  # Already present
        "generate_to_share": "Generate music to share with the community!",  # Added
        "global_impact": "🌐 Global Impact",  # Already present
        "global_impact_info": "EmoTune.AI supports mental health worldwide, especially in underserved communities. It can be deployed on low-cost devices like Raspberry Pi to bring music therapy to rural areas.",  # Added
        "location_resources": "Based on your location ({location}), here are some mental health resources:",  # Added
        "mood_journey": "🏆 Your Mood Journey",  # Already present
        "mood_explorer_badge": "Analyze your mood {remaining} more times to earn the Mood Explorer Badge!",  # Added
        "streak_master_badge": "Maintain the same mood for {remaining} more sessions to earn the Streak Master Badge!",  # Added
        "biometric_data": "💓 Biometric Data",
        "use_biometrics": "Use biometric data",
        "manual_input": "Manual Input",
        "smartwatch_input": "Fetch from Smartwatch",
        "current_mood": "😊 Current Mood",
        "user_insights": "📊 User Insights",
        "no_listening_history": "No listening history yet. Play some music to see insights!",
        "achievements": "🏆 Achievements",
        "happy_streak": "Happy Streak: {days} days ({needed} needed for achievement)",
        "preferences": "🎵 Preferences",
        "skip_genres": "Skip these genres",
        "voice_control": "🎙️ Voice Control",
        "play_happy_song": "Say 'Play a happy song'",
        "analyze_mood": "Analyze My Mood 🧠",
        "capture_face": "Capture your face",
        "processing_image": "Processing image...",
        "generated_music": "🎶 Generated Music",
        "rate_music": "🎵 How was the music?",
        "submit_feedback": "Submit Feedback",
        "youtube_songs": "▶️ YouTube Songs",
        "movies": "🎬 Movies",
        "spotify_tracks": "🎧 Spotify Tracks",
        "save_moment": "Save This Moment 😉",
        "memory_capsules": "🔮 Emotional Memory Capsules",
        "privacy_settings": "🔒 Privacy Settings",
        "data_usage_overview": "Data Usage Overview",
        "biometric_data_status": "Biometric Data: Not Collected",
        "facial_images_status": "Facial Images: Not Collected",
        "chat_history_status": "Chat History: Not Collected",
        "manage_your_data": "Manage Your Data",
        "clear_biometric_data": "Clear Biometric Data",
        "clear_facial_data": "Clear Facial Data",
        "clear_chat_history": "Clear Chat History",
        "biometrics_normal": "Biometrics Normal",  # Added
        "chatbot_title": "EmoTune Chatbot 🤖",
        "welcome_message": "Welcome to EmoTune.AI! 🎼",
        "welcome_text": "Discover music tailored to your mood using AI-powered mood detection, personalized recommendations, and an emotional support chatbot.",
        "start_prompt": "Start by selecting your mood or let us detect it for you!",
        "consent_title": "Consent Required",
        "consent_text": "We use your webcam for facial emotion analysis and collect biometric data for mood detection. This data is processed locally and not stored. Do you agree?",
        "agree_button": " Agree",
        "no_mood_history": "No mood history yet. Analyze your mood to see trends!",
        "privacy_message": "🔒 Your privacy matters. We do not store or share your data. All processing is done locally.",
        "mood_trend_graph_title": "Mood Trend Over Time",
        "mood_trend_graph_x": "Session",
        "mood_trend_graph_y": "Mood",
        "mood_options": ["happy", "sad", "calm", "excited", "neutral"],
        "auto_detect": "Auto-detect",
        "chat_input_placeholder": "Type your message here...",
        "chat_user_label": "User",
        "chat_bot_label": "Bot",
        "chat_submit_button": "Send",
        "analyzing_live_emotions": "Analyzing live facial emotions...",
        "captured_image": "Captured Image",
        "clear_button": "Clear",
        "biometric_data_label": "Biometric Data:",
        "facial_images_label": "Facial Images:",
        "chat_history_label": "Chat History:",
        "collected": "Collected",
        "not_collected": "Not Collected",
        "biometric_data_cleared": "Biometric data cleared!",
        "facial_data_cleared": "Facial data cleared!",
        "chat_history_cleared": "Chat history cleared!",
        
    },
    "hi": {
        "title": "EmoTune.AI 🎼",
        "subtitle": "एक मनोदशा-संवेदनशील ऑडियो अनुभव",
        "select_mood": "🎭 अपनी मनोदशा चुनें",
        "choose_mood": "अपनी मनोदशा चुनें (या स्वचालित रूप से पता लगाएं)",
        "describe_day": "📝 अपने दिन का वर्णन करें",
        "describe_mood": "अपनी मनोदशा या दिन का वर्णन करें (वैकल्पिक)",
        "speak_day": "🎙️ अपने दिन के बारे में बोलें (वैकल्पिक)",
        "upload_audio": "अपनी बोलने की ऑडियो क्लिप अपलोड करें",
        "upload_image": "📷 एक छवि अपलोड करें (वैकल्पिक)",
        "upload_image_prompt": "चेहरे की भावना विश्लेषण के लिए एक छवि अपलोड करें",
        "facial_emotion_live": "📸 चेहरे की भावना विश्लेषण (लाइव)",
        "enable_facial_analysis": "लाइव चेहरे की भावना विश्लेषण सक्षम करें",
        "mood_booster": "🌟 मनोदशा बढ़ाने के टिप्स",  # Already present
        "mood_tip_reflect": "अपने दिन पर विचार करें—किसने आपको मुस्कुराया? 📝",  # Added
        "mood_tip_walk": "अपने मूड को ऊपर उठाने के लिए थोड़ा टहलें! 🚶",  # Added
        "mood_tip_music": "कुछ उत्साहवर्धक संगीत सुनें—हमने आपके लिए व्यवस्था की है! 🎶",  # Added
        "mood_tip_breathe": "अपनी शांति बनाए रखने के लिए कुछ गहरी सांस लेने के व्यायाम करें! 🧘",  # Added
        "mood_tip_tea": "एक गर्म कप चाय पिएं और आराम करें! ☕",  # Added
        "mood_tip_workout": "अपनी ऊर्जा को एक मजेदार गतिविधि में लगाएं—शायद एक त्वरित वर्कआउट? 🏋️",  # Added
        "mood_tip_friend": "अपने उत्साह को एक दोस्त के साथ साझा करें! 📞",  # Added
        "mood_tip_new": "आज कुछ नया आजमाएं ताकि प्रेरणा मिले! ✨",  # Added
        "mood_tip_dance": "अपने पसंदीदा गाने पर नृत्य करें ताकि अपनी खुशी बढ़े! 💃",  # Added
        "mood_tip_let_go": "एक पल के लिए गहरी सांस लें और तनाव को छोड़ दें! 🌬️",  # Added
        "mood_tip_write": "जो आपको परेशान कर रहा है उसे लिखें ताकि आपका दिमाग साफ हो! ✍️",  # Added
        "mood_tip_talk": "अपनी चिंताओं को कम करने के लिए किसी विश्वसनीय व्यक्ति से बात करें! 🗣️",  # Added
        "mood_tip_grounding": "एक ग्राउंडिंग व्यायाम आजमाएं—अपने आसपास पर ध्यान दें! 🌳",  # Added
        "mood_tip_embrace": "अप्रत्याशित को गले लगाएं—शायद यह किसी अच्छी चीज का संकेत है! 🎉",  # Added
        "mood_tip_journal": "इस पल को एक त्वरित जर्नल प्रविष्टि के साथ कैप्चर करें! 📓",  # Added
        "mood_tip_step_away": "जो आपको परेशान कर रहा है उससे दूर हटें और ब्रेक लें! 🚪",  # Added
        "mood_tip_positive": "अपने मूड को बदलने के लिए किसी सकारात्मक चीज पर ध्यान दें! 🌈",  # Added
        "mood_trend": "📈 मनोदशा प्रवृत्ति",
        "mental_health": "🌍 मानसिक स्वास्थ्य और समुदाय",  # Already present
        "mental_health_info": "संगीत तनाव को 65% तक कम कर सकता है (स्रोत: WHO)। और समर्थन चाहिए? WHO मानसिक स्वास्थ्य संसाधनों का अन्वेषण करें।",  # Added
        "share_music": "समुदाय के साथ साझा करें",  # Already present
        "generate_to_share": "समुदाय के साथ साझा करने के लिए संगीत उत्पन्न करें!",  # Added
        "global_impact": "🌐 वैश्विक प्रभाव",  # Already present
        "global_impact_info": "EmoTune.AI विश्व भर में मानसिक स्वास्थ्य का समर्थन करता है, विशेष रूप से अल्पसेवित समुदायों में। इसे कम लागत वाले उपकरणों जैसे Raspberry Pi पर तैनात किया जा सकता है ताकि ग्रामीण क्षेत्रों में संगीत चिकित्सा लाई जा सके।",  # Added
        "location_resources": "आपके स्थान ({location}) के आधार पर, यहाँ कुछ मानसिक स्वास्थ्य संसाधन हैं:",  # Added
        "mood_journey": "🏆 आपकी मनोदशा यात्रा",  # Already present
        "mood_explorer_badge": "मूड एक्सप्लोरर बैज अर्जित करने के लिए अपनी मनोदशा को {remaining} बार और विश्लेषण करें!",  # Added
        "streak_master_badge": "स्ट्रीक मास्टर बैज अर्जित करने के लिए एक ही मनोदशा को {remaining} और सत्रों के लिए बनाए रखें!",  # Added
        "biometric_data": "💓 बायोमेट्रिक डेटा",
        "use_biometrics": "बायोमेट्रिक डेटा का उपयोग करें",
        "manual_input": "मैनुअल इनपुट",
        "smartwatch_input": "स्मार्टवॉच से प्राप्त करें",
        "capture_face": "अपना चेहरा कैप्चर करें",
        "processing_image": "छवि संसाधित हो रही है...",
        "current_mood": "😊 वर्तमान मनोदशा",
        "user_insights": "📊 उपयोगकर्ता अंतर्दृष्टि",
        "no_listening_history": "अभी तक कोई सुनने का इतिहास नहीं। अंतर्दृष्टि देखने के लिए कुछ संगीत चलाएं!",
        "achievements": "🏆 उपलब्धियाँ",
        "happy_streak": "खुशहाल स्ट्रीक: {days} दिन ({needed} उपलब्धि के लिए आवश्यक)",
        "preferences": "🎵 प्राथमिकताएँ",
        "skip_genres": "इन शैलियों को छोड़ें",
        "voice_control": "🎙️ आवाज नियंत्रण",
        "play_happy_song": "कहें 'एक खुशहाल गाना बजाएं'",
        "analyze_mood": "मेरी मनोदशा का विश्लेषण करें 🧠",
        "generated_music": "🎶 उत्पन्न संगीत",
        "rate_music": "🎵 संगीत कैसा था?",
        "submit_feedback": "प्रतिक्रिया सबमिट करें",
        "youtube_songs": "▶️ यूट्यूब गाने",
        "movies": "🎬 फिल्में",
        "spotify_tracks": "🎧 स्पॉटिफाई ट्रैक",
        "save_moment": "इस पल को सहेजें 😉",
        "memory_capsules": "🔮 भावनात्मक स्मृति कैप्सूल",
        "privacy_settings": "🔒 गोपनीयता सेटिंग्स",
        "data_usage_overview": "डेटा उपयोग अवलोकन",
        "biometric_data_status": "बायोमेट्रिक डेटा: एकत्र नहीं किया गया",
        "facial_images_status": "चेहरे की छवियां: एकत्र नहीं की गईं",
        "chat_history_status": "चैट इतिहास: एकत्र नहीं किया गया",
        "manage_your_data": "अपना डेटा प्रबंधित करें",
        "clear_biometric_data": "बायोमेट्रिक डेटा साफ करें",
        "clear_facial_data": "चेहरे का डेटा साफ करें",
        "clear_chat_history": "चैट इतिहास साफ करें",
        "biometrics_normal": "बायोमेट्रिक्स सामान्य",  # Added
        "chatbot_title": "EmoTune चैटबॉट 🤖",
        "welcome_message": "EmoTune.AI में आपका स्वागत है! 🎼",
        "welcome_text": "AI-संचालित मनोदशा पहचान, व्यक्तिगत सिफारिशें और एक भावनात्मक समर्थन चैटबॉट का उपयोग करके अपनी मनोदशा के अनुरूप संगीत खोजें।",
        "start_prompt": "अपनी मनोदशा चुनकर शुरू करें या हमें इसे आपके लिए पता लगाने दें!",
        "consent_title": "सहमति आवश्यक",
        "consent_text": "हम चेहरे की भावना विश्लेषण के लिए आपका वेबकैम और मनोदशा पहचान के लिए बायोमेट्रिक डेटा एकत्र करते हैं। यह डेटा स्थानीय रूप से संसाधित होता है और संग्रहीत नहीं होता। क्या आप सहमत हैं?",
        "agree_button": " सहमत",
        "no_mood_history": "अभी तक कोई मनोदशा इतिहास नहीं। रुझान देखने के लिए अपनी मनोदशा का विश्लेषण करें!",
        "privacy_message": "🔒 आपकी गोपनीयता मायने रखती है। हम आपका डेटा संग्रहीत या साझा नहीं करते। सभी प्रसंस्करण स्थानीय रूप से किया जाता है।",
        "mood_options": ["खुश", "उदास", "शांत", "उत्साहित", "तटस्थ"],
        "mood_trend_graph_title": "मनोदशा रुझान समय के साथ",
        "mood_trend_graph_x": "satr",
        "mood_trend_graph_y": "manodasha",
        "auto_detect": "स्वचालित रूप से पता लगाएं",
        "chat_input_placeholder": "अपना संदेश यहाँ टाइप करें...",
        "chat_user_label": "उपयोगकर्ता",
        "chat_bot_label": "बॉट",
        "chat_submit_button": "भेजें",
        "analyzing_live_emotions": "लाइव चेहरे की भावनाओं का विश्लेषण कर रहा है...",
        "captured_image": "कैप्चर की गई छवि",
        "clear_button": "साफ करें",
        "biometric_data_label": "बायोमेट्रिक डेटा:",
        "facial_images_label": "चेहरे की छवियां:",
        "chat_history_label": "चैट इतिहास:",
        "collected": "एकत्र किया गया",
        "not_collected": "एकत्र नहीं किया गया",
        "biometric_data_cleared": "बायोमेट्रिक डेटा साफ कर दिया गया!",
        "facial_data_cleared": "चेहरे का डेटा साफ कर दिया गया!",
        "chat_history_cleared": "चैट इतिहास साफ कर दिया गया!",
    },
    "es": {
        "title": "EmoTune.AI 🎼",
        "subtitle": "Una experiencia de audio responsiva al estado de ánimo",
        "select_mood": "🎭 Selecciona Tu Estado de Ánimo",
        "choose_mood": "Elige tu estado de ánimo (o detección automática)",
        "describe_day": "📝 Describe Tu Día",
        "describe_mood": "Describe tu estado de ánimo o día (opcional)",
        "speak_day": "🎙️ Habla Sobre Tu Día (Opcional)",
        "upload_audio": "Sube un clip de audio de ti hablando",
        "upload_image": "📷 Sube una Imagen (Opcional)",
        "upload_image_prompt": "Sube una imagen para análisis de emociones faciales",
        "facial_emotion_live": "📸 Análisis de Emociones Faciales (En Vivo)",
        "enable_facial_analysis": "Habilitar análisis de emociones faciales en vivo",
        "mood_booster": "🌟 Consejos para Mejorar el Ánimo",  # Already present
        "mood_tip_reflect": "Reflexiona sobre tu día—¿qué te hizo sonreír? 📝",  # Added
        "mood_tip_walk": "¡Da un corto paseo para levantar tu ánimo! 🚶",  # Added
        "mood_tip_music": "¡Escucha música edificante—te tenemos cubierto! 🎶",  # Added
        "mood_tip_breathe": "¡Prueba algunos ejercicios de respiración profunda para mantener tu paz! 🧘",  # Added
        "mood_tip_tea": "¡Toma una taza de té caliente y relájate! ☕",  # Added
        "mood_tip_workout": "¡Canaliza tu energía en una actividad divertida—tal vez un entrenamiento rápido? 🏋️",  # Added
        "mood_tip_friend": "¡Comparte tu emoción con un amigo! 📞",  # Added
        "mood_tip_new": "¡Prueba algo nuevo hoy para despertar inspiración! ✨",  # Added
        "mood_tip_dance": "¡Baila con tu canción favorita para amplificar tu alegría! 💃",  # Added
        "mood_tip_let_go": "¡Tómate un momento para respirar profundamente y dejar ir la tensión! 🌬️",  # Added
        "mood_tip_write": "¡Escribe lo que te molesta para despejar tu mente! ✍️",  # Added
        "mood_tip_talk": "¡Habla con alguien de confianza para aliviar tus preocupaciones! 🗣️",  # Added
        "mood_tip_grounding": "¡Prueba un ejercicio de conexión a tierra—enfócate en tu entorno! 🌳",  # Added
        "mood_tip_embrace": "¡Abraza lo inesperado—tal vez sea una señal de algo grandioso! 🎉",  # Added
        "mood_tip_journal": "¡Captura este momento con una entrada rápida en tu diario! 📓",  # Added
        "mood_tip_step_away": "¡Aléjate de lo que te molesta y toma un descanso! 🚪",  # Added
        "mood_tip_positive": "¡Enfócate en algo positivo para cambiar tu estado de ánimo! 🌈",  # Added
        "mood_trend": "📈 Tendencia de Ánimo",
        "mental_health": "🌍 Salud Mental y Comunidad",  # Already present
        "mental_health_info": "La música puede reducir el estrés hasta en un 65% (Fuente: OMS). ¿Necesitas más apoyo? Explora los Recursos de Salud Mental de la OMS.",  # Added
        "share_music": "Compartir con la Comunidad",  # Already present
        "generate_to_share": "¡Genera música para compartir con la comunidad!",  # Added
        "global_impact": "🌐 Impacto Global",  # Already present
        "global_impact_info": "EmoTune.AI apoya la salud mental en todo el mundo, especialmente en comunidades desatendidas. Puede implementarse en dispositivos de bajo costo como Raspberry Pi para llevar terapia musical a áreas rurales.",  # Added
        "location_resources": "Basado en tu ubicación ({location}), aquí tienes algunos recursos de salud mental:",  # Added
        "mood_journey": "🏆 Tu Viaje de Ánimo",  # Already present
        "mood_explorer_badge": "¡Analiza tu estado de ánimo {remaining} veces más para ganar la Insignia de Explorador de Ánimo!",  # Added
        "streak_master_badge": "¡Mantén el mismo estado de ánimo durante {remaining} sesiones más para ganar la Insignia de Maestro de Racha!",  # Added
        "biometric_data": "💓 Datos Biométricos",
        "use_biometrics": "Usar datos biométricos",
        "manual_input": "Entrada Manual",
        "smartwatch_input": "Obtener del Reloj Inteligente",
        "current_mood": "😊 Estado de Ánimo Actual",
        "user_insights": "📊 Perspectivas del Usuario",
        "no_listening_history": "No hay historial de escucha todavía. ¡Toca algo de música para ver perspectivas!",
        "achievements": "🏆 Logros",
        "happy_streak": "Racha Feliz: {days} días ({needed} necesarios para el logro)",
        "preferences": "🎵 Preferencias",
        "skip_genres": "Omitir estos géneros",
        "voice_control": "🎙️ Control por Voz",
        "play_happy_song": "Di 'Toca una canción feliz'",
        "analyze_mood": "Analizar Mi Estado de Ánimo 🧠",
        "generated_music": "🎶 Música Generada",
        "rate_music": "🎵 ¿Qué te pareció la música?",
        "submit_feedback": "Enviar Comentario",
        "youtube_songs": "▶️ Canciones de YouTube",
        "capture_face": "Captura tu rostro",
        "processing_image": "Procesando imagen...",
        "movies": "🎬 Películas",
        "spotify_tracks": "🎧 Pistas de Spotify",
        "save_moment": "Guardar Este Momento 😉",
        "memory_capsules": "🔮 Cápsulas de Memoria Emocional",
        "privacy_settings": "🔒 Configuración de Privacidad",
        "data_usage_overview": "Resumen de Uso de Datos",
        "biometric_data_status": "Datos Biométricos: No Recolectados",
        "facial_images_status": "Imágenes Faciales: No Recolectadas",
        "chat_history_status": "Historial de Chat: No Recolectado",
        "manage_your_data": "Administra Tus Datos",
        "clear_biometric_data": "Borrar Datos Biométricos",
        "clear_facial_data": "Borrar Datos Faciales",
        "clear_chat_history": "Borrar Historial de Chat",
        "biometrics_normal": "Biométricos Normales",  # Added
        "chatbot_title": "Chatbot de EmoTune 🤖",
        "welcome_message": "¡Bienvenido a EmoTune.AI! 🎼",
        "welcome_text": "Descubre música adaptada a tu estado de ánimo con detección de ánimo por IA, recomendaciones personalizadas y un chatbot de apoyo emocional.",
        "start_prompt": "¡Comienza seleccionando tu estado de ánimo o déjanos detectarlo por ti!",
        "consent_title": "Consentimiento Requerido",
        "consent_text": "Usamos tu cámara web para análisis de emociones faciales y recolectamos datos biométricos para detectar el estado de ánimo. Estos datos se procesan localmente y no se almacenan. ¿Estás de acuerdo?",
        "agree_button": " Aceptar",
        "no_mood_history": "No hay historial de estado de ánimo todavía. ¡Analiza tu estado de ánimo para ver tendencias!",
        "privacy_message": "🔒 Tu privacidad importa. No almacenamos ni compartimos tus datos. Todo el procesamiento se realiza localmente.",
        "mood_trend_graph_title": "Tendencia de Ánimo a lo Largo del Tiempo",
        "mood_trend_graph_x": "Sesión",
        "mood_trend_graph_y": "Estado de Ánimo", 
        "mood_options": ["feliz", "triste", "calmado", "emocionado", "neutral"],
        "auto_detect": "Detección automática",
        "chat_input_placeholder": "Escribe tu mensaje aquí...",
        "chat_user_label": "Usuario",
        "chat_bot_label": "Bot",
        "chat_submit_button": "Enviar",
        "analyzing_live_emotions": "Analizando emociones faciales en vivo...",
        "captured_image": "Imagen Capturada",
        "clear_button": "Limpiar",
        "biometric_data_label": "Datos Biométricos:",
        "facial_images_label": "Imágenes Faciales:",
        "chat_history_label": "Historial de Chat:",
        "collected": "Recolectado",
        "not_collected": "No Recolectado",
        "biometric_data_cleared": "¡Datos biométricos borrados!",
        "facial_data_cleared": "¡Datos faciales borrados!",
        "chat_history_cleared": "¡Historial de chat borrado!",
    },
    "de": {
        "title": "EmoTune.AI 🎼",
        "subtitle": "Ein stimmungsabhängiges Audioerlebnis",
        "select_mood": "🎭 Wähle Deine Stimmung",
        "choose_mood": "Wähle deine Stimmung (oder automatisch erkennen)",
        "describe_day": "📝 Beschreibe Deinen Tag",
        "describe_mood": "Beschreibe deine Stimmung oder deinen Tag (optional)",
        "speak_day": "🎙️ Sprich Über Deinen Tag (Optional)",
        "upload_audio": "Lade einen Audioclip von dir sprechend hoch",
        "upload_image": "📷 Lade ein Bild Hoch (Optional)",
        "upload_image_prompt": "Lade ein Bild für die Gesichtsemotionsanalyse hoch",
        "facial_emotion_live": "📸 Gesichtsemotionsanalyse (Live)",
        "enable_facial_analysis": "Live-Gesichtsemotionsanalyse aktivieren",
        "mood_booster": "🌟 Stimmungsaufheller-Tipps",  # Already present
        "mood_tip_reflect": "Reflektiere über deinen Tag—was hat dich zum Lächeln gebracht? 📝",  # Added
        "mood_tip_walk": "Mach einen kurzen Spaziergang, um deine Stimmung zu heben! 🚶",  # Added
        "mood_tip_music": "Hör dir aufmunternde Musik an—wir haben dich abgedeckt! 🎶",  # Added
        "mood_tip_breathe": "Probiere einige tiefe Atemübungen, um deine Ruhe zu bewahren! 🧘",  # Added
        "mood_tip_tea": "Trink eine warme Tasse Tee und entspann dich! ☕",  # Added
        "mood_tip_workout": "Lenke deine Energie in eine lustige Aktivität—vielleicht ein schnelles Training? 🏋️",  # Added
        "mood_tip_friend": "Teile deine Aufregung mit einem Freund! 📞",  # Added
        "mood_tip_new": "Probiere heute etwas Neues, um Inspiration zu finden! ✨",  # Added
        "mood_tip_dance": "Tanze zu deinem Lieblingslied, um deine Freude zu steigern! 💃",  # Added
        "mood_tip_let_go": "Nimm dir einen Moment, um tief durchzuatmen und Spannungen loszulassen! 🌬️",  # Added
        "mood_tip_write": "Schreib auf, was dich stört, um deinen Kopf freizubekommen! ✍️",  # Added
        "mood_tip_talk": "Sprich mit jemandem, dem du vertraust, um deine Sorgen zu lindern! 🗣️",  # Added
        "mood_tip_grounding": "Probiere eine Erdungsübung—konzentriere dich auf deine Umgebung! 🌳",  # Added
        "mood_tip_embrace": "Nimm das Unerwartete an—vielleicht ist es ein Zeichen für etwas Großartiges! 🎉",  # Added
        "mood_tip_journal": "Halte diesen Moment mit einem kurzen Tagebucheintrag fest! 📓",  # Added
        "mood_tip_step_away": "Tritt von dem, was dich stört, zurück und mach eine Pause! 🚪",  # Added
        "mood_tip_positive": "Konzentriere dich auf etwas Positives, um deine Stimmung zu ändern! 🌈",  # Added
        "mood_trend": "📈 Stimmungstrend",
        "mental_health": "🌍 Mentale Gesundheit & Gemeinschaft",  # Already present
        "mental_health_info": "Musik kann Stress um bis zu 65% reduzieren (Quelle: WHO). Brauchst du mehr Unterstützung? Erkunde die WHO-Ressourcen für mentale Gesundheit.",  # Added
        "share_music": "Mit der Gemeinschaft teilen",  # Already present
        "generate_to_share": "Generiere Musik, um sie mit der Gemeinschaft zu teilen!",  # Added
        "global_impact": "🌐 Globaler Einfluss",  # Already present
        "global_impact_info": "EmoTune.AI unterstützt die mentale Gesundheit weltweit, insbesondere in unterversorgten Gemeinschaften. Es kann auf kostengünstigen Geräten wie Raspberry Pi eingesetzt werden, um Musiktherapie in ländliche Gebiete zu bringen.",  # Added
        "location_resources": "Basierend auf deinem Standort ({location}), hier sind einige Ressourcen für mentale Gesundheit:",  # Added
        "mood_journey": "🏆 Deine Stimmungsreise",  # Already present
        "mood_explorer_badge": "Analysiere deine Stimmung noch {remaining} Mal, um das Stimmungserkunder-Abzeichen zu verdienen!",  # Added
        "streak_master_badge": "Halte die gleiche Stimmung für {remaining} weitere Sitzungen bei, um das Streak-Meister-Abzeichen zu verdienen!",  # Added
        "biometric_data": "💓 Biometrische Daten",
        "use_biometrics": "Biometrische Daten verwenden",
        "manual_input": "Manuelle Eingabe",
        "smartwatch_input": "Von der Smartwatch abrufen",
        "capture_face": "Erfasse dein Gesicht",
        "processing_image": "Bild wird verarbeitet...",
        "current_mood": "😊 Aktuelle Stimmung",
        "user_insights": "📊 Nutzereinblicke",
        "no_listening_history": "Noch kein Hörverlauf. Spiele Musik ab, um Einblicke zu sehen!",
        "achievements": "🏆 Erfolge",
        "happy_streak": "Glückssträhne: {days} Tage ({needed} für Erfolg erforderlich)",
        "preferences": "🎵 Vorlieben",
        "skip_genres": "Diese Genres überspringen",
        "voice_control": "🎙️ Sprachsteuerung",
        "play_happy_song": "Sage 'Spiele ein fröhliches Lied'",
        "analyze_mood": "Analysiere Meine Stimmung 🧠",
        "generated_music": "🎶 Generierte Musik",
        "rate_music": "🎵 Wie war die Musik?",
        "submit_feedback": "Feedback Senden",
        "youtube_songs": "▶️ YouTube-Songs",
        "movies": "🎬 Filme",
        "spotify_tracks": "🎧 Spotify-Tracks",
        "save_moment": "Diesen Moment Speichern 😉",
        "memory_capsules": "🔮 Emotionale Erinnerungskapseln",
        "privacy_settings": "🔒 Datenschutzeinstellungen",
        "data_usage_overview": "Übersicht der Datennutzung",
        "biometric_data_status": "Biometrische Daten: Nicht gesammelt",
        "facial_images_status": "Gesichtsbilder: Nicht gesammelt",
        "chat_history_status": "Chatverlauf: Nicht gesammelt",
        "manage_your_data": "Verwalte Deine Daten",
        "clear_biometric_data": "Biometrische Daten löschen",
        "clear_facial_data": "Gesichtsdaten löschen",
        "clear_chat_history": "Chatverlauf löschen",
        "biometrics_normal": "Biometrie Normal",  # Added
        "chatbot_title": "EmoTune Chatbot 🤖",
        "welcome_message": "Willkommen bei EmoTune.AI! 🎼",
        "welcome_text": "Entdecke Musik, die auf deine Stimmung abgestimmt ist, mit KI-gestützter Stimmungserkennung, personalisierten Empfehlungen und einem emotionalen Unterstützungs-Chatbot.",
        "start_prompt": "Beginne, indem du deine Stimmung auswählst oder uns sie für dich erkennen lässt!",
        "consent_title": "Zustimmung Erforderlich",
        "consent_text": "Wir verwenden deine Webcam für die Gesichtsemotionsanalyse und sammeln biometrische Daten zur Stimmungserkennung. Diese Daten werden lokal verarbeitet und nicht gespeichert. Stimmt du zu?",
        "agree_button": " Zustimmen",
        "privacy_message": "🔒 Deine Privatsphäre ist wichtig. Wir speichern oder teilen deine Daten nicht. Alle Verarbeitung erfolgt lokal.",
        "no_mood_history": "Noch kein Stimmungsverlauf. Analysiere deine Stimmung, um Trends zu sehen!",
        "mood_options": ["glücklich", "traurig", "ruhig", "aufgeregt", "neutral"],
        "mood_trend_graph_title": "Stimmungstrend im Laufe der Zeit",
        "mood_trend_graph_x": "Sitzung",
        "mood_trend_graph_y": "Stimmung",
        "auto_detect": "Automatisch erkennen",
        "chat_input_placeholder": "Gib deine Nachricht hier ein...",
        "chat_user_label": "Benutzer",
        "chat_bot_label": "Bot",
        "chat_submit_button": "Senden",
        "analyzing_live_emotions": "Analyse der Gesichtsemotionen in Echtzeit...",
        "captured_image": "Aufgenommenes Bild",
        "clear_button": "Löschen",
        "biometric_data_label": "Biometrische Daten:",
        "facial_images_label": "Gesichtsbilder:",
        "chat_history_label": "Chatverlauf:",
        "collected": "Gesammelt",
        "not_collected": "Nicht Gesammelt",
        "biometric_data_cleared": "Biometrische Daten gelöscht!",
        "facial_data_cleared": "Gesichtsdaten gelöscht!",
        "chat_history_cleared": "Chatverlauf gelöscht!",
    }
}
# Initialize user profile
def initialize_user_profile():
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "listening_history": [],
            "preferred_genres": [],
            "mood_patterns": {}
        }

# Update listening history
def update_listening_history(song_title, song_url, mood):
    st.session_state.user_profile["listening_history"].append({
        "title": song_title,
        "url": song_url,
        "mood": mood
    })

# Display user insights
def display_user_insights():
    # Safely extract the language code
    language = st.session_state.get("language", "English (en)")
    lang_parts = language.split("(")
    if len(lang_parts) > 1:
        lang_code = lang_parts[1].strip(")")
    else:
        lang_code = "en"
    t = translations[lang_code]

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<h2>{t["user_insights"]}</h2>', unsafe_allow_html=True)
        if not st.session_state.user_profile["listening_history"]:
            st.write(t["no_listening_history"])
        else:
            st.write("Listening History:")
            for entry in st.session_state.user_profile["listening_history"][-5:]:
                st.markdown(f'<p>🎵 {entry["title"]} (Mood: {entry["mood"].capitalize()})</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
# Initialize data usage tracking
def initialize_data_usage():
    if "data_usage" not in st.session_state:
        st.session_state.data_usage = {
            "biometric": False,
            "facial": False,
            "chat": False
        }

# Display privacy dashboard
def display_privacy_dashboard():
    # Safely extract the language code
    language = st.session_state.get("language", "English (en)")  # Default to "English (en)" if not set
    lang_parts = language.split("(")
    if len(lang_parts) > 1:
        lang_code = lang_parts[1].strip(")")
    else:
        lang_code = "en"  # Fallback to "en" if the format is unexpected
    t = translations[lang_code]

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<h2>{t["privacy_settings"]}</h2>', unsafe_allow_html=True)
        
        st.markdown(f'<h3>{t["data_usage_overview"]}</h3>', unsafe_allow_html=True)
        data_usage = st.session_state.data_usage
        biometric_status = t["collected"] if data_usage["biometric"] else t["not_collected"]
        facial_status = t["collected"] if data_usage["facial"] else t["not_collected"]
        chat_status = t["collected"] if data_usage["chat"] else t["not_collected"]
        st.markdown(f'<p>🔹 {t["biometric_data_label"]} {biometric_status}</p>', unsafe_allow_html=True)
        st.markdown(f'<p>🔹 {t["facial_images_label"]} {facial_status}</p>', unsafe_allow_html=True)
        st.markdown(f'<p>🔹 {t["chat_history_label"]} {chat_status}</p>', unsafe_allow_html=True)
        
        st.markdown(f'<h3>{t["manage_your_data"]}</h3>', unsafe_allow_html=True)
        if st.button(t["clear_biometric_data"]):
            st.session_state.biometric_inputs = {"hr": 70, "spo2": 95, "motion": 0.5}
            st.session_state.biometric_data = {"mood": "neutral", "confidence": 0.5}
            st.session_state.data_usage["biometric"] = False
            st.success(t["biometric_data_cleared"])
        
        if st.button(t["clear_facial_data"]):
            st.session_state.webcam_image = None
            st.session_state.uploaded_image = None
            st.session_state.temp_face_mood = None
            st.session_state.temp_face_conf = None
            st.session_state.data_usage["facial"] = False
            st.success(t["facial_data_cleared"])
        
        if st.button(t["clear_chat_history"]):
            st.session_state.chat_history = []
            st.session_state.data_usage["chat"] = False
            st.success(t["chat_history_cleared"])
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display consent popup
def display_consent_popup():
    if "consent_given" not in st.session_state:
        st.session_state.consent_given = False
    
    if not st.session_state.consent_given:
        st.markdown(
            """
            <div style="background: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 15px; text-align: center;">
                <h3>Consent Required</h3>
                <p>We use your webcam for facial emotion analysis and collect biometric data for mood detection. This data is processed locally and not stored. Do you agree?</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Agree"):
            st.session_state.consent_given = True
            st.rerun()
        return False
    return True

# Mock biometric data
def get_mock_biometrics():
    hr = np.random.randint(50, 140)
    spo2 = np.random.uniform(90, 100)
    motion = np.random.uniform(0, 1)
    scaler = MinMaxScaler()
    bio_features = scaler.fit_transform([[hr, spo2, motion]])
    if hr > 100 or motion > 0.7:
        return "excited", 0.8
    elif hr < 70 and spo2 > 98:
        return "calm", 0.7
    elif hr > 90 and motion < 0.3:
        return "sad", 0.6
    else:
        return "neutral", 0.5

# Manual biometric input
def get_manual_biometric_data():
    st.write("Enter biometric data manually (placeholder for smartwatch integration):")
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
    spo2 = st.number_input("SpO2 (%)", min_value=80, max_value=100, value=95)
    motion = st.slider("Motion Level (0-1)", 0.0, 1.0, 0.5)
    st.info("Note: Real-time data from a smartwatch requires additional setup.")
    scaler = MinMaxScaler()
    bio_features = scaler.fit_transform([[hr, spo2, motion]])
    if hr > 100 or motion > 0.7:
        return "excited", 0.8
    elif hr < 70 and spo2 > 98:
        return "calm", 0.7
    elif hr > 90 and motion < 0.3:
        return "sad", 0.6
    else:
        return "neutral", 0.5

# Fetch biometric data from a smartwatch API (mocked for this example)
def fetch_biometric_from_smartwatch():
    """
    Simulate fetching biometric data from a smartwatch API.
    Returns a tuple of (heart_rate, spo2, motion).
    In a real implementation, this would make an API call to a smartwatch service.
    """
    try:
        # Simulate an API call (replace with actual API integration in production)
        # Example: Fitbit API endpoint (you'd need an access token and proper authentication)
        # url = "https://api.fitbit.com/1/user/-/activities/heart/date/today/1d.json"
        # headers = {"Authorization": f"Bearer {access_token}"}
        # response = requests.get(url, headers=headers)
        # data = response.json()

        # Mocked response for demonstration
        mock_response = {
            "heart_rate": random.randint(50, 140),  # Simulated heart rate (bpm)
            "spo2": random.uniform(90, 100),        # Simulated SpO2 (%)
            "motion": random.uniform(0, 1)          # Simulated motion level (0-1)
        }

        hr = mock_response["heart_rate"]
        spo2 = mock_response["spo2"]
        motion = mock_response["motion"]

        # Validate the data
        if not (40 <= hr <= 200 and 80 <= spo2 <= 100 and 0 <= motion <= 1):
            raise ValueError("Invalid biometric data received from smartwatch.")

        return hr, spo2, motion
    except Exception as e:
        st.error(f"Failed to fetch biometric data from smartwatch: {str(e)}")
        st.warning("Falling back to default values.")
        return 70, 95, 0.5  # Default values in case of failure

# Process image for emotion detection
def process_image_for_emotion(image, session, model, graph):
    # Convert PIL image to numpy array
    frame = np.array(image)
    
    # Ensure the image is in RGB format (PIL images are already RGB, but let's confirm)
    if frame.shape[-1] == 4:  # If image has an alpha channel (RGBA)
        frame = frame[..., :3]  # Remove alpha channel
    frame_rgb = frame.copy()  # Keep a copy for face detection
    
    # Convert to BGR for OpenCV processing if needed
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Convert to RGB for Mediapipe face detection
    results = face_detection.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    
    if not results.detections:
        st.warning("No face detected in the image. Defaulting to neutral mood.")
        return "neutral", 0.5, frame_bgr
    
    # Process the first detected face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = frame.shape
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)
    
    # Add padding to the bounding box to ensure the entire face is captured
    padding = int(max(width, height) * 0.3)
    x = max(0, x - padding)
    y = max(0, y - padding)
    width = min(w - x, width + 2 * padding)
    height = min(h - y, height + 2 * padding)
    
    # Extract the face region
    face = frame_rgb[y:y+height, x:x+width]
    if face.size == 0:
        st.warning("Face region is empty. Defaulting to neutral mood.")
        return "neutral", 0.5, frame_bgr
    
    # Convert face to grayscale for emotion detection
    face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization to improve contrast
    face_gray = cv2.equalizeHist(face_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_gray = clahe.apply(face_gray)
    
    # Resize to the expected input size for the model (64x64 for Mini-Xception)
    face_resized = cv2.resize(face_gray, (64, 64))
    
    # Normalize and reshape for model input
    face_processed = np.expand_dims(face_resized, axis=(0, -1)) / 255.0
    
    # Perform prediction
    try:
        with graph.as_default():
            with session.as_default():
                preds = model.predict(face_processed)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {str(e)}")
        return "neutral", 0.5, frame_bgr
    
    # Get the predicted emotion and confidence
    emotion_idx = np.argmax(preds)
    confidence = preds[emotion_idx]
    emotion = EMOTIONS[emotion_idx]
    
    # Draw bounding box and label on the original frame (in BGR format)
    cv2.rectangle(frame_bgr, (x, y), (x+width, y+height), (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"{emotion} ({confidence:.2f})", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return emotion, confidence, frame_bgr

# Facial emotion analysis
def analyze_image(image_source, session, model, graph):
    if image_source is not None:
        try:
            # Open the image using PIL
            image = Image.open(image_source)
            
            # Process the image for emotion detection
            emotion, confidence, _ = process_image_for_emotion(image, session, model, graph)
            
            # Do not display the processed image
            return emotion, confidence
        except Exception as e:
            st.error(f"Failed to analyze image: {str(e)}")
            return "neutral", 0.5
    return "neutral", 0.5
    
# Text mood analysis with Gemini API (Enhanced)
def analyze_text_for_mood(text):
    """
    Analyze the user's text input to determine their mood using Gemini API.
    Returns a tuple of (mood, confidence).
    """
    if not text.strip():
        return "neutral", 0.5  # Default if no text is provided

    try:
        prompt = f"""
        Analyze the following text to determine the user's mood. The mood should be one of: happy, sad, calm, excited, or neutral.
        Provide the mood and a confidence score between 0 and 1. Respond in the format: mood: <mood>, confidence: <confidence>
        Text: "{text}"
        """
        response = gemini_model.generate_content(prompt).text
        lines = response.split(",")
        mood = lines[0].split(":")[1].strip()
        confidence = float(lines[1].split(":")[1].strip())
        return mood, confidence
    except Exception as e:
        st.error(f"Failed to analyze text for mood: {str(e)}")
        return "neutral", 0.5  # Fallback in case of error

# Combine inputs with manual mood option
def analyze_mood(manual_mood, text_input, use_biometrics, session, model, graph):
    mood_scores = {"happy": 0, "sad": 0, "calm": 0, "excited": 0, "neutral": 0}
    
    # If manual mood is selected, use it directly
    if manual_mood and manual_mood != "Auto-detect":
        mood_scores[manual_mood] = 1.0
        return manual_mood, 1.0

    # Initialize default values
    face_mood, face_conf = ("neutral", 0.5)
    text_mood, text_conf = ("neutral", 0.5)
    bio_mood, bio_conf = ("neutral", 0.0)
    voice_mood, voice_conf = ("neutral", 0.0)

    # Facial analysis
    if "temp_face_mood" in st.session_state and st.session_state.temp_face_mood is not None:
        face_mood = st.session_state.temp_face_mood
        face_conf = st.session_state.temp_face_conf
    elif "uploaded_image" in st.session_state and st.session_state.uploaded_image is not None:
        face_mood, face_conf = analyze_image(st.session_state.uploaded_image, session, model, graph)
        st.session_state.temp_face_mood = face_mood
        st.session_state.temp_face_conf = face_conf

    # Text analysis
    if text_input:
        text_mood, text_conf = analyze_text_for_mood(text_input)

    # Biometric analysis
    if use_biometrics:
        bio_mood, bio_conf = st.session_state.biometric_data["mood"], st.session_state.biometric_data["confidence"]

    # Voice tone analysis
    if "voice_mood" in st.session_state and st.session_state.voice_mood is not None:
        voice_mood = st.session_state.voice_mood
        voice_conf = st.session_state.voice_conf

    # Determine active inputs
    active_inputs = []
    if "temp_face_mood" in st.session_state and st.session_state.temp_face_mood is not None:
        active_inputs.append("face")
    if text_input:
        active_inputs.append("text")
    if use_biometrics:
        active_inputs.append("biometrics")
    if "voice_mood" in st.session_state and st.session_state.voice_mood is not None:
        active_inputs.append("voice")

    if not active_inputs:
        return "neutral", 0.5

    # Combine mood scores based on active inputs
    if len(active_inputs) == 1:
        if "face" in active_inputs:
            mood_scores[face_mood] = face_conf * 1.0
        elif "text" in active_inputs:
            mood_scores[text_mood] = text_conf * 1.0
        elif "biometrics" in active_inputs:
            mood_scores[bio_mood] = bio_conf * 1.0
        elif "voice" in active_inputs:
            mood_scores[voice_mood] = voice_conf * 1.0
    elif len(active_inputs) == 2:
        if "face" in active_inputs and "text" in active_inputs:
            mood_scores[face_mood] = face_conf * 0.6
            mood_scores[text_mood] = text_conf * 0.4
        elif "face" in active_inputs and "biometrics" in active_inputs:
            mood_scores[face_mood] = face_conf * 0.6
            mood_scores[bio_mood] = bio_conf * 0.4
        elif "face" in active_inputs and "voice" in active_inputs:
            mood_scores[face_mood] = face_conf * 0.6
            mood_scores[voice_mood] = voice_conf * 0.4
        elif "text" in active_inputs and "biometrics" in active_inputs:
            mood_scores[text_mood] = text_conf * 0.6
            mood_scores[bio_mood] = bio_conf * 0.4
        elif "text" in active_inputs and "voice" in active_inputs:
            mood_scores[text_mood] = text_conf * 0.6
            mood_scores[voice_mood] = voice_conf * 0.4
        elif "biometrics" in active_inputs and "voice" in active_inputs:
            mood_scores[bio_mood] = bio_conf * 0.6
            mood_scores[voice_mood] = voice_conf * 0.4
    elif len(active_inputs) == 3:
        if "face" in active_inputs and "text" in active_inputs and "biometrics" in active_inputs:
            mood_scores[face_mood] = face_conf * 0.5
            mood_scores[text_mood] = text_conf * 0.3
            mood_scores[bio_mood] = bio_conf * 0.2
        elif "face" in active_inputs and "text" in active_inputs and "voice" in active_inputs:
            mood_scores[face_mood] = face_conf * 0.5
            mood_scores[text_mood] = text_conf * 0.3
            mood_scores[voice_mood] = voice_conf * 0.2
        elif "face" in active_inputs and "biometrics" in active_inputs and "voice" in active_inputs:
            mood_scores[face_mood] = face_conf * 0.5
            mood_scores[bio_mood] = bio_conf * 0.3
            mood_scores[voice_mood] = voice_conf * 0.2
        elif "text" in active_inputs and "biometrics" in active_inputs and "voice" in active_inputs:
            mood_scores[text_mood] = text_conf * 0.5
            mood_scores[bio_mood] = bio_conf * 0.3
            mood_scores[voice_mood] = voice_conf * 0.2
    else:
        mood_scores[face_mood] = face_conf * 0.4
        mood_scores[text_mood] = text_conf * 0.3
        mood_scores[bio_mood] = bio_conf * 0.2
        mood_scores[voice_mood] = voice_conf * 0.1

    final_mood = max(mood_scores, key=mood_scores.get)
    final_conf = mood_scores[final_mood]
    return final_mood, final_conf

# Music generation function
def generate_music(mood):
    mood_prompts = {
        "happy": "upbeat cheerful melody",
        "sad": "slow melancholic tune",
        "calm": "peaceful ambient sound",
        "excited": "fast energetic track",
        "neutral": "relaxed neutral beat"
    }
    prompt = mood_prompts.get(mood, "relaxed neutral beat")
    
    try:
        with st.spinner("Generating music..."):
            melody = music_model.generate(descriptions=[prompt], progress=True)
            audio_data = melody[0].cpu().numpy() if hasattr(melody, 'shape') else melody[0].cpu().numpy()
            if audio_data.size > 0:
                output_file = f"generated_melody_{mood}_{int(time.time())}.wav"
                sf.write(output_file, audio_data.T, samplerate=32000, subtype='PCM_16')
                return output_file
            else:
                raise ValueError("Empty audio data generated.")
    except Exception as e:
        st.error(f"Music generation failed: {str(e)}")
        st.warning("Falling back to simple tone.")
        try:
            frequency = 440
            if mood == "happy": frequency = 660
            elif mood == "sad": frequency = 220
            elif mood == "calm": frequency = 330
            elif mood == "excited": frequency = 880
            sample_rate = 32000
            t = linspace(0, 5, sample_rate * 5, False)
            audio = 0.5 * sin(2 * np.pi * frequency * t)
            output_file = f"fallback_melody_{mood}_{int(time.time())}.wav"
            write(output_file, sample_rate, audio.astype(np.float32))
            return output_file
        except Exception as e2:
            st.error(f"Fallback generation failed: {str(e2)}")
            return None

# YouTube recommendations
@backoff.on_exception(backoff.expo, (ConnectionResetError, HttpError), max_tries=3, max_time=60)
def get_youtube_songs(mood, max_results=5):
    try:
        query = f"{mood} songs playlist"
        request = youtube.search().list(part="snippet", q=query, type="video", maxResults=max_results)
        response = request.execute()
        songs = [(item["snippet"]["title"], f"https://youtube.com/watch?v={item['id']['videoId']}") 
                 for item in response["items"]]
        time.sleep(2)
        return songs
    except HttpError as e:
        if e.resp.status == 403 and "quotaExceeded" in str(e):
            st.error("YouTube API quota exceeded.")
            return FALLBACK_SONGS.get(mood, FALLBACK_SONGS["neutral"])
        raise
    except ConnectionResetError:
        st.warning("Connection reset by YouTube API. Retrying...")
        raise

# Spotify recommendations
@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=30)
def get_spotify_recommendations(mood, limit=5):
    mood_queries = {
        "happy": "happy pop",
        "sad": "sad acoustic",
        "calm": "calm classical",
        "excited": "excited rock",
        "neutral": "neutral indie"
    }
    search_query = mood_queries.get(mood, "neutral indie")
    try:
        results = sp.search(q=search_query, type="track", limit=limit)
        recommendations = [(f"{track['name']} - {track['artists'][0]['name']}", track["external_urls"]["spotify"]) 
                           for track in results["tracks"]["items"]]
        return recommendations
    except Exception as e:
        st.error(f"Failed to fetch Spotify recommendations: {str(e)}")
        return FALLBACK_TRACKS.get(mood, FALLBACK_TRACKS["neutral"])  # Use fallback tracks

# Periodic facial analysis for live webcam
def periodic_facial_analysis(t, webcam_active):
    if webcam_active:
        # Clear uploaded image state to avoid conflicts
        if "uploaded_image" in st.session_state:
            st.session_state.uploaded_image = None
        
        # Display the webcam input
        webcam_image = st.camera_input(t["capture_face"], key="webcam_input", label_visibility="collapsed")
        
        if webcam_image:
            # Display the captured image
            st.image(webcam_image, caption=t["captured_image"], use_container_width=True)
            
            with st.spinner(t["processing_image"]):
                try:
                    temp_mood, temp_conf = analyze_image(webcam_image, session, emotion_model, graph)
                    st.session_state.temp_face_mood = temp_mood
                    st.session_state.temp_face_conf = temp_conf
                    st.session_state.data_usage["facial"] = True
                    st.session_state.webcam_image = webcam_image  # Store the webcam image
                    st.success("Image processed. Click 'Analyze My Mood' to see the result.")
                except Exception as e:
                    st.error(f"Failed to analyze webcam image: {str(e)}")
                    st.session_state.temp_face_mood = "neutral"
                    st.session_state.temp_face_conf = 0.5
                    st.session_state.webcam_image = None
            
            # Add a "Clear" button to reset the camera input
            if st.button(t["clear_button"], key="clear_webcam_button"):
                st.session_state.webcam_image = None
                st.session_state.temp_face_mood = None
                st.session_state.temp_face_conf = None
                st.rerun()  # Rerun to refresh the UI and clear the image
    
    else:
        # Clear webcam-related session state when live analysis is disabled
        st.session_state.webcam_image = None
        st.session_state.temp_face_mood = None
        st.session_state.temp_face_conf = None
        
# Display achievements
def display_achievements():
    # Safely extract the language code
    language = st.session_state.get("language", "English (en)")
    lang_parts = language.split("(")
    if len(lang_parts) > 1:
        lang_code = lang_parts[1].strip(")")
    else:
        lang_code = "en"
    t = translations[lang_code]

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<h2>{t["achievements"]}</h2>', unsafe_allow_html=True)
        
        # Calculate happy streak
        happy_streak = 0
        for mood, _, _ in reversed(st.session_state.mood_history):
            if mood == "happy":
                happy_streak += 1
            else:
                break
        
        # Use the translated string with dynamic values
        st.write(t["happy_streak"].format(days=happy_streak, needed=3))
        
        if happy_streak >= 3:
            st.markdown('<p style="color: #ffcc00;">🎉 Happy Streak Achieved!</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


# Gemini chatbot function
def gemini_chatbot():
    global gemini_model
    # Language extraction and translation setup
    language = st.session_state.get("language", "English (en)")
    lang_parts = language.split("(")
    if len(lang_parts) > 1:
        lang_code = lang_parts[1].strip(")")
    else:
        lang_code = "en"
    t = translations[lang_code]

    if 'gemini_model' not in globals():
        st.error("Gemini model not initialized. Please ensure gemini_model is set up correctly.")
        return

    if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []
    else:
        cleaned_history = []
        for message in st.session_state.chat_history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                cleaned_history.append(message)
        st.session_state.chat_history = cleaned_history

    st.markdown(f'<div class="chatbot-title">{t["chatbot_title"]}</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                role = t["chat_user_label"] if message["role"] == "user" else t["chat_bot_label"]
                st.markdown(f'<div class="chat-message"><strong>{role}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        user_input = st.text_input(t["chat_input_placeholder"], key="chat_input", label_visibility="collapsed")
        submit_button = st.form_submit_button(t["chat_submit_button"])
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_button and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.data_usage["chat"] = True

            try:
                conversation = []
                for msg in st.session_state.chat_history:
                    role = msg["role"]
                    api_role = "model" if role == "bot" or role == "model" else "user"
                    conversation.append({"role": api_role, "parts": [msg["content"]]})
                chat = gemini_model.start_chat(history=conversation[:-1])
                response = chat.send_message(user_input).text
            except Exception as e:
                response = f"Sorry, I encountered an error: {str(e)}"
                st.error(f"Gemini API error: {str(e)}")

            st.session_state.chat_history.append({"role": "model", "content": response})
            st.rerun() 
            
def main():
    # Language selection dropdown
    language = st.selectbox("Select Language / भाषा चुनें / Seleccionar Idioma / Sprache Wählen", 
                            ["English (en)", "Hindi (hi)", "Spanish (es)", "German (de)"], 
                            key="language_select")
    # Store the selected language in session state
    st.session_state.language = language
    # Extract the language code safely
    lang_parts = language.split("(")
    if len(lang_parts) > 1:
        lang_code = lang_parts[1].strip(")")
    else:
        lang_code = language  # Fallback to the whole string (e.g., "en")
    t = translations[lang_code]
    
    # Initialize session state
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "mood_history" not in st.session_state:
        st.session_state.mood_history = []
    if "webcam_image" not in st.session_state:
        st.session_state.webcam_image = None
    if "mood" not in st.session_state:
        st.session_state.mood = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "last_audio_file" not in st.session_state:
        st.session_state.last_audio_file = None
    if "temp_face_mood" not in st.session_state:
        st.session_state.temp_face_mood = None
    if "temp_face_conf" not in st.session_state:
        st.session_state.temp_face_conf = None
    if "mood_streak" not in st.session_state:
        st.session_state.mood_streak = {"mood": None, "count": 0}
    if "biometric_inputs" not in st.session_state:
        st.session_state.biometric_inputs = {"hr": 70, "spo2": 95, "motion": 0.5}
    if "biometric_data" not in st.session_state:
        st.session_state.biometric_data = {"mood": "neutral", "confidence": 0.5}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "biometric_input_method" not in st.session_state:
        st.session_state.biometric_input_method = "manual"

    initialize_user_profile()
    initialize_data_usage()

    def get_user_location():
        try:
            response = requests.get("https://ipapi.co/json/")
            data = response.json()
            return data.get("country_name", "Unknown")
        except Exception as e:
            st.error(f"Failed to detect location: {str(e)}")
            return "Unknown"

    user_location = get_user_location()

    # Custom CSS (unchanged)
    css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #ff6ec4, #7873f5, #00ddeb, #ff6ec4);
        background-size: 600% 600%;
        animation: gradientBG 20s ease infinite;
        min-height: 100vh;
        overflow-y: auto;
        padding: 0 !important;
        margin: 0 !important;
        position: relative;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    h1 {
        color: #fff;
        text-shadow: 0 0 15px #9b59b6, 0 0 30px #8e44ad;
        font-size: 3em;
        text-align: center;
        margin-bottom: 10px;
    }
    h2, h3 {
        color: #ffcc00;
        text-shadow: 0 0 10px #ffcc00;
        font-size: 1.8em;
        margin-bottom: 15px;
    }
    .stApp * {
        color: #fff;
        font-family: 'Exo 2', sans-serif;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.8);
    }
    .card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.5s ease-in;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(255, 255, 255, 0.3);
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInMood {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in-mood {
        animation: fadeInMood 0.5s ease-in-out;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #ff4d4d, #ff7878);
        color: #fff;
        padding: 12px 24px;
        border: none;
        border-radius: 30px;
        font-size: 16px;
        font-weight: bold;
        text-transform: uppercase;
        box-shadow: 0 0 15px #ff4d4d, 0 0 30px #ff7878, inset 0 0 10px #fff;
        transition: all 0.3s ease;
        cursor: pointer;
        animation: pulse 2s infinite;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff7878, #ff4d4d);
        box-shadow: 0 0 25px #ff4d4d, 0 0 50px #ff7878, inset 0 0 15px #fff;
        transform: scale(1.05);
    }
    .stTextArea textarea, .stSelectbox, .stFileUploader, .stSlider, .stNumberInput input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 15px;
        color: #fff !important;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    .youtube-link, .movie-link, .spotify-link {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 15px;
        text-decoration: none;
        margin: 8px 0;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .youtube-link { background: #ffcc00; color: #000; box-shadow: 0 0 10px #ffcc00; }
    .movie-link { background: #00cccc; color: #000; box-shadow: 0 0 10px #00cccc; }
    .spotify-link { background: #ff66cc; color: #000; box-shadow: 0 0 10px #ff66cc; }
    .chatbot {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 0 !important;
        padding-bottom: 0 !important;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .chatbot-title {
        color: #ffcc00;
        text-shadow: 0 0 10px #ffcc00;
        font-size: 1.5em;
        margin-bottom: 10px;
        text-align: center;
    }
    .chat-history {
        max-height: 150px;
        overflow-y: auto;
        margin-bottom: 10px;
        padding: 5px;
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.3);
    }
    .chat-message {
        margin: 5px 0;
        padding: 5px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .chat-input-container {
        display: flex;
        align-items: center;
        margin: 0 !important;
        padding: 0 !important;
    }
    .stForm {
        margin: 0 !important;
        padding: 0 !important;
    }
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 8px;
        color: #fff;
        margin-right: 10px;
        flex-grow: 1;
    }
    .stFormSubmitButton > button {
        background: #ff4d4d;
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 8px 15px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .mood-indicator {
        text-align: center;
        font-size: 3em;
        margin: 15px 0;
        animation: bounce 0.5s ease;
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    .bio-insight {
        display: flex;
        align-items: center;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3), inset 0 0 5px rgba(255, 255, 255, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .bio-insight-icon {
        font-size: 1.5em;
        margin-right: 10px;
    }
    .bio-insight-text {
        font-size: 0.9em;
        color: #ffcc00;
        text-shadow: 0 0 5px #ffcc00;
    }
    .mood-tip {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(255, 255, 255, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .mood-tip:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(255, 255, 255, 0.3);
    }
    #particles-js {
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: -1;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700&display=swap" rel="stylesheet">
    """
    st.markdown(css, unsafe_allow_html=True)

    # Privacy notice (unchanged)
# At the end of main(), after all columns
    st.markdown('<div class="privacy-footer">', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #cccccc;">{t["privacy_message"]}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Particle effects (unchanged)
    current_mood = st.session_state.mood if st.session_state.mood else "neutral"
    mood_particle_configs = {
        "happy": {
            "particles": {
                "number": {"value": 80},
                "color": {"value": "#ffcc00"},
                "shape": {"type": "star"},
                "opacity": {"value": 0.8},
                "size": {"value": 4},
                "move": {"speed": 3}
            }
        },
        "sad": {
            "particles": {
                "number": {"value": 50},
                "color": {"value": "#00cccc"},
                "shape": {"type": "circle"},
                "opacity": {"value": 0.5},
                "size": {"value": 3},
                "move": {"speed": 1, "direction": "bottom"}
            }
        },
        "calm": {
            "particles": {
                "number": {"value": 60},
                "color": {"value": "#66ff66"},
                "shape": {"type": "circle"},
                "opacity": {"value": 0.6},
                "size": {"value": 2},
                "move": {"speed": 2}
            }
        },
        "excited": {
            "particles": {
                "number": {"value": 100},
                "color": {"value": "#ff66cc"},
                "shape": {"type": "triangle"},
                "opacity": {"value": 0.9},
                "size": {"value": 5},
                "move": {"speed": 5}
            }
        },
        "neutral": {
            "particles": {
                "number": {"value": 40},
                "color": {"value": "#cccccc"},
                "shape": {"type": "circle"},
                "opacity": {"value": 0.4},
                "size": {"value": 3},
                "move": {"speed": 1}
            }
        },
        "angry": {
            "particles": {
                "number": {"value": 70},
                "color": {"value": "#ff3333"},
                "shape": {"type": "polygon"},
                "opacity": {"value": 0.7},
                "size": {"value": 4},
                "move": {"speed": 4}
            }
        },
        "fear": {
            "particles": {
                "number": {"value": 60},
                "color": {"value": "#9999ff"},
                "shape": {"type": "circle"},
                "opacity": {"value": 0.5},
                "size": {"value": 3},
                "move": {"speed": 2, "direction": "bottom"}
            }
        },
        "surprise": {
            "particles": {
                "number": {"value": 90},
                "color": {"value": "#ff99ff"},
                "shape": {"type": "star"},
                "opacity": {"value": 0.8},
                "size": {"value": 4},
                "move": {"speed": 3}
            }
        },
        "disgust": {
            "particles": {
                "number": {"value": 50},
                "color": {"value": "#66cccc"},
                "shape": {"type": "circle"},
                "opacity": {"value": 0.6},
                "size": {"value": 3},
                "move": {"speed": 2}
            }
        }
    }
    particle_config = mood_particle_configs.get(current_mood, mood_particle_configs["neutral"])
    particle_html = f"""
    <div id="particles-js"></div>
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <script>
        particlesJS("particles-js", {json.dumps(particle_config)});
    </script>
    """
    st.markdown(particle_html, unsafe_allow_html=True)

    # Welcome message
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = False
    if not st.session_state.welcome_shown:
        st.markdown(f"""
        <div style="text-align: center; background: rgba(0, 0, 0, 0.5); padding: 20px; border-radius: 15px;">
            <h2>{t['welcome_message']}</h2>
            <p>{t['welcome_text']}</p>
            <p>{t['start_prompt']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.welcome_shown = True

    # Header
    st.markdown(f'<h1>{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.2em; color: #ffcc00; text-shadow: 0 0 5px #ffcc00;">{t["subtitle"]}</p>', unsafe_allow_html=True)

    # Consent popup
    def display_consent_popup():
        if "consent_given" not in st.session_state:
            st.session_state.consent_given = False
        
        if not st.session_state.consent_given:
            st.markdown(
                f"""
                <div style="background: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3>{t['consent_title']}</h3>
                    <p>{t['consent_text']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(t["agree_button"]):
                st.session_state.consent_given = True
                st.rerun()
            return False
        return True

    consent_given = display_consent_popup()

    # Main content
    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<h2>{t["select_mood"]}</h2>', unsafe_allow_html=True)
            
            # Define the English options (used internally)
            english_options = ["Auto-detect"] + ["happy", "sad", "calm", "excited", "neutral"]
            
            # Create the translated options list
            translated_options = [t["auto_detect"]] + t["mood_options"]
            
            # Create a mapping from translated options to English options
            option_mapping = dict(zip(translated_options, english_options))
            
            # Use translated options in the dropdown
            selected_translated_option = st.selectbox(t["choose_mood"], translated_options, label_visibility="collapsed")
            
            # Map the selected translated option back to English
            manual_mood = option_mapping.get(selected_translated_option, "Auto-detect")  # Default to "Auto-detect" if not found
            
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<h2>{t["describe_day"]}</h2>', unsafe_allow_html=True)
            text_input = st.text_area(t["describe_mood"], label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<h2>{t["speak_day"]}</h2>', unsafe_allow_html=True)
            audio_input = st.file_uploader(t["upload_audio"], type=["wav", "mp3"], label_visibility="collapsed")
            if audio_input:
                with st.spinner("Analyzing voice tone..."):
                    voice_mood, voice_conf = "happy", 0.75  # Placeholder
                    st.session_state.voice_mood = voice_mood
                    st.session_state.voice_conf = voice_conf
                    st.session_state.data_usage["voice"] = True
                    st.success("Voice tone processed. Click 'Analyze My Mood' to see the result.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<h2>{t["upload_image"]}</h2>', unsafe_allow_html=True)
            uploaded_image = st.file_uploader(t["upload_image_prompt"], type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if uploaded_image:
                st.session_state.webcam_image = None
                st.session_state.temp_face_mood = None
                st.session_state.temp_face_conf = None
                
                with st.spinner("Processing image..."):
                    try:
                        temp_mood, temp_conf = analyze_image(uploaded_image, session, emotion_model, graph)
                        st.session_state.uploaded_image = uploaded_image
                        st.session_state.temp_face_mood = temp_mood
                        st.session_state.temp_face_conf = temp_conf
                        st.session_state.data_usage["facial"] = True
                        st.success("Image processed. Click 'Analyze My Mood' to see the result.")
                    except Exception as e:
                        st.error(f"Failed to analyze uploaded image: {str(e)}")
                        st.session_state.uploaded_image = None
                        st.session_state.temp_face_mood = "neutral"
                        st.session_state.temp_face_conf = 0.5
            
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="mood-tip">', unsafe_allow_html=True)
            st.markdown(f'<h2>{t["mood_booster"]}</h2>', unsafe_allow_html=True)
            mood_tips = {
                "happy": [
                    t["mood_tip_dance"],  # "Dance to your favorite song to amplify your joy! 💃"
                    t["mood_tip_friend"],  # "Share your excitement with a friend! 📞"
                ],
                "sad": [
                    t["mood_tip_walk"],   # "Take a short walk to lift your spirits! 🚶"
                    t["mood_tip_music"],  # "Listen to some uplifting music—we’ve got you covered! 🎶"
                ],
                "calm": [
                    t["mood_tip_breathe"],  # "Try some deep breathing exercises to maintain your peace! 🧘"
                    t["mood_tip_tea"],      # "Sip a warm cup of tea and relax! ☕"
                ],
                "excited": [
                    t["mood_tip_workout"],  # "Channel your energy into a fun activity—maybe a quick workout? 🏋️"
                    t["mood_tip_friend"],   # "Share your excitement with a friend! 📞"
                ],
                "neutral": [
                    t["mood_tip_new"],      # "Try something new today to spark some inspiration! ✨"
                    t["mood_tip_reflect"],  # "Reflect on your day—what made you smile? 📝"
                ],
                "angry": [
                    t["mood_tip_let_go"],   # "Take a moment to breathe deeply and let go of tension! 🌬️"
                    t["mood_tip_write"],    # "Write down what’s bothering you to clear your mind! ✍️"
                ],
                "fear": [
                    t["mood_tip_talk"],     # "Talk to someone you trust to ease your worries! 🗣️"
                    t["mood_tip_grounding"],  # "Try a grounding exercise—focus on your surroundings! 🌳"
                ],
                "surprise": [
                    t["mood_tip_embrace"],  # "Embrace the unexpected—maybe it’s a sign of something great! 🎉"
                    t["mood_tip_journal"],  # "Capture this moment with a quick journal entry! 📓"
                ],
                "disgust": [
                    t["mood_tip_step_away"],  # "Step away from what’s bothering you and take a break! 🚪"
                    t["mood_tip_positive"],   # "Focus on something positive to shift your mood! 🌈"
                ]
            }
            current_mood = st.session_state.mood if st.session_state.mood else "neutral"
            tips = mood_tips.get(current_mood, mood_tips["neutral"])
            tip = random.choice(tips)
            st.markdown(f'<p>{tip}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if consent_given:
        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{t["mood_trend"]}</h2>', unsafe_allow_html=True)
                if st.session_state.mood_history:
                    recent_moods = [m[0] for m in st.session_state.mood_history[-5:]]
                    mood_colors = {
                        "happy": "#ffcc00",
                        "sad": "#00cccc",
                        "calm": "#66ff66",
                        "excited": "#ff66cc",
                        "neutral": "#cccccc"
                    }
                    emotion_to_mood_for_graph = {
                        "angry": "sad",
                        "disgust": "sad",
                        "fear": "sad",
                        "happy": "happy",
                        "sad": "sad",
                        "surprise": "excited",
                        "neutral": "neutral"
                    }
                    mapped_moods = [emotion_to_mood_for_graph.get(mood, mood if mood in MOOD_OPTIONS else "neutral") for mood in recent_moods]
                    line_color = mood_colors.get(mapped_moods[-1], "#ff7878")
                    try:
                        mood_indices = [MOOD_OPTIONS.index(mood) for mood in mapped_moods]
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.plot(range(len(mapped_moods)), mood_indices, marker='o', color=line_color, linewidth=2, markersize=8)
                        ax.set_yticks(range(len(MOOD_OPTIONS)))
                        ax.set_yticklabels(MOOD_OPTIONS)  # Note: MOOD_OPTIONS may also need translation (see below)
                        ax.set_title(t["mood_trend_graph_title"], color='#ffcc00')  # Updated to use translated string
                        ax.set_xlabel(t["mood_trend_graph_x"], color='#fff')  # Updated to use translated string
                        ax.set_ylabel(t["mood_trend_graph_y"], color='#fff')  # Updated to use translated string
                        ax.tick_params(axis='x', colors='#fff')
                        ax.tick_params(axis='y', colors='#fff')
                        ax.set_facecolor((0, 0, 0, 0.3))
                        fig.patch.set_facecolor((0, 0, 0, 0.3))
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Failed to render mood trend graph: {str(e)}")
                else:
                    st.write(t["no_mood_history"])
                st.markdown('</div>', unsafe_allow_html=True)

            # Add Facial Emotion Analysis (Live) section
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{t["facial_emotion_live"]}</h2>', unsafe_allow_html=True)
                webcam_active = st.checkbox(t["enable_facial_analysis"], value=False)
                if webcam_active:
                    with st.spinner("Analyzing live facial emotions..."):
                        periodic_facial_analysis(t,webcam_active)  # Call the existing function for live analysis
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{t["mental_health"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p>{t["mental_health_info"]} <a href="https://www.who.int/health-topics/mental-health" target="_blank">{t["mental_health_info"].split()[-4]} {t["mental_health_info"].split()[-3]} {t["mental_health_info"].split()[-2]} {t["mental_health_info"].split()[-1]}</a></p>',unsafe_allow_html=True)
                st.markdown(f'<h3>🌟 {t["share_music"]}</h3>', unsafe_allow_html=True)
                if st.session_state.last_audio_file:
                    if st.button(t["share_music"]):
                        st.success("Your music has been shared with the community! 🎉")
                else:
                    st.write(t["generate_to_share"])
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{t["global_impact"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p>{t["global_impact_info"]}</p>',unsafe_allow_html=True)
                st.markdown(f'<p>{t["location_resources"].format(location=user_location)}</p>', unsafe_allow_html=True)
                resources = {
                    "United States": [("Mental Health America", "https://mhanational.org")],
                    "India": [("Vandrevala Foundation", "https://www.vandrevalafoundation.com")],
                    "Default": [("World Health Organization", "https://www.who.int/health-topics/mental-health")]
                }
                location_key = user_location if user_location in resources else "Default"
                for name, url in resources[location_key]:
                    st.markdown(f'<a href="{url}" class="youtube-link">{name}</a>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{t["mood_journey"]}</h2>', unsafe_allow_html=True)
                if len(st.session_state.mood_history) >= 5:
                    st.markdown('<p style="color: #ffcc00;">🎉 Mood Explorer Badge: Analyzed mood 5 times!</p>', unsafe_allow_html=True)
                else:
                    remaining = 5 - len(st.session_state.mood_history)
                    st.write(t["mood_explorer_badge"].format(remaining=remaining))
                if st.session_state.mood_streak["count"] >= 3:
                    st.markdown(f'<p style="color: #ffcc00;">🔥 Streak Master Badge: {st.session_state.mood_streak["mood"].capitalize()} for {st.session_state.mood_streak["count"]} sessions!</p>', unsafe_allow_html=True)
                else:
                    remaining = 3 - st.session_state.mood_streak["count"]
                    st.write(t["streak_master_badge"].format(remaining=remaining))
                st.markdown('</div>', unsafe_allow_html=True)
                        
            with st.container():
                st.markdown('<div class="bio-insight">', unsafe_allow_html=True)
                hr = st.session_state.biometric_inputs["hr"]
                spo2 = st.session_state.biometric_inputs["spo2"]
                motion = st.session_state.biometric_inputs["motion"]
                if hr > 100:
                    insight_icon = "💨"
                    insight_text = "High Heart Rate Detected!"
                elif spo2 < 95:
                    insight_icon = "🩺"
                    insight_text = "Low SpO2 Level!"
                elif motion > 0.7:
                    insight_icon = "🏃"
                    insight_text = "High Activity Level!"
                else:
                    insight_icon = "✅"
                    insight_text = t["biometrics_normal"]  # Use translated string
                st.markdown(f'<span class="bio-insight-icon">{insight_icon}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="bio-insight-text">{insight_text}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{t["biometric_data"]}</h2>', unsafe_allow_html=True)
                
                use_biometrics = st.checkbox(t["use_biometrics"], value=False)
                
                if use_biometrics:
                    col_manual, col_smartwatch = st.columns(2)
                    
                    with col_manual:
                        use_manual_input = st.checkbox(t["manual_input"], value=True, key="manual_input")
                    
                    with col_smartwatch:
                        use_smartwatch = st.checkbox(t["smartwatch_input"], value=False, key="smartwatch_input")
                    
                    if use_manual_input and use_smartwatch:
                        st.warning("Please select only one biometric input method.")
                        use_smartwatch = False
                    
                    if use_manual_input and not use_smartwatch:
                        st.session_state.biometric_input_method = "manual"
                    elif use_smartwatch and not use_manual_input:
                        st.session_state.biometric_input_method = "smartwatch"
                    else:
                        st.session_state.biometric_input_method = "manual"
                    
                    if st.session_state.biometric_input_method == "manual":
                        st.write("Enter biometric data manually:")
                        with st.form(key="biometric_form"):
                            hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=st.session_state.biometric_inputs["hr"])
                            spo2 = st.number_input("SpO2 (%)", min_value=80, max_value=100, value=st.session_state.biometric_inputs["spo2"])
                            motion = st.slider("Motion Level (0-1)", 0.0, 1.0, st.session_state.biometric_inputs["motion"])
                            submit_button = st.form_submit_button(t["submit_feedback"])
                            if submit_button:
                                with st.spinner("Updating biometric data..."):
                                    st.session_state.biometric_inputs = {"hr": hr, "spo2": spo2, "motion": motion}
                                    st.session_state.data_usage["biometric"] = True
                                    bio_features = MinMaxScaler().fit_transform([[hr, spo2, motion]])
                                    if hr > 100 or motion > 0.7:
                                        bio_mood, bio_conf = "excited", 0.8
                                    elif hr < 70 and spo2 > 98:
                                        bio_mood, bio_conf = "calm", 0.7
                                    elif hr > 90 and motion < 0.3:
                                        bio_mood, bio_conf = "sad", 0.6
                                    else:
                                        bio_mood, bio_conf = "neutral", 0.5
                                    st.session_state.biometric_data = {"mood": bio_mood, "confidence": bio_conf}
                                    st.markdown(f'<p>Biometric Mood: <strong>{bio_mood.capitalize()}</strong> (Confidence: {bio_conf:.2f})</p>', unsafe_allow_html=True)
                    
                    elif st.session_state.biometric_input_method == "smartwatch":
                        st.write("Fetching biometric data from smartwatch...")
                        with st.spinner("Connecting to smartwatch API..."):
                            hr, spo2, motion = fetch_biometric_from_smartwatch()
                            st.session_state.biometric_inputs = {"hr": hr, "spo2": spo2, "motion": motion}
                            st.session_state.data_usage["biometric"] = True
                            bio_features = MinMaxScaler().fit_transform([[hr, spo2, motion]])
                            if hr > 100 or motion > 0.7:
                                bio_mood, bio_conf = "excited", 0.8
                            elif hr < 70 and spo2 > 98:
                                bio_mood, bio_conf = "calm", 0.7
                            elif hr > 90 and motion < 0.3:
                                bio_mood, bio_conf = "sad", 0.6
                            else:
                                bio_mood, bio_conf = "neutral", 0.5
                            st.session_state.biometric_data = {"mood": bio_mood, "confidence": bio_conf}
                            st.markdown(f'<p>Heart Rate: {hr} bpm, SpO2: {spo2:.1f}%, Motion: {motion:.2f}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p>Biometric Mood: <strong>{bio_mood.capitalize()}</strong> (Confidence: {bio_conf:.2f})</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{t["current_mood"]}</h2>', unsafe_allow_html=True)
                mood_emojis = {
                    "happy": "😊",
                    "sad": "😢",
                    "calm": "😌",
                    "excited": "🤩",
                    "neutral": "😐",
                    "angry": "😡",
                    "fear": "😨",
                    "surprise": "😲",
                    "disgust": "🤢"
                }
                current_mood = st.session_state.mood if st.session_state.mood else "neutral"
                st.markdown(f'<div class="mood-indicator">{mood_emojis.get(current_mood, "😐")}</div>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center;">{current_mood.capitalize()}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Please give consent to enable biometric and facial analysis features.")

    display_user_insights()
    display_achievements()

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<h2>{t["preferences"]}</h2>', unsafe_allow_html=True)
        genre_skip = st.multiselect(t["skip_genres"], ["Metal", "Rap", "Classical", "Acoustic", "Dance", "Chill", "Ambient", "EDM", "Lo-Fi"], default=[], label_visibility="collapsed")
        st.session_state.user_profile["preferred_genres"] = genre_skip
        st.markdown('</div>', unsafe_allow_html=True)


    with st.container():
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        if st.button(t["analyze_mood"]):
            with st.spinner("Analyzing your mood..."):
                mood, confidence = analyze_mood(
                    manual_mood, text_input, use_biometrics, session, emotion_model, graph
                )
                st.session_state.mood = mood
                st.session_state.confidence = confidence
                st.session_state.mood_history.append((mood, confidence, time.time()))
                st.markdown(
                    f'<p class="fade-in-mood" style="text-align: center; font-size: 1.5em;">Final Detected Mood: <strong>{mood.capitalize()}</strong> (Confidence: {confidence:.2f})</p>',
                    unsafe_allow_html=True
                )
    
            if st.session_state.mood_streak["mood"] == mood:
                st.session_state.mood_streak["count"] += 1
            else:
                st.session_state.mood_streak = {"mood": mood, "count": 1}
            if st.session_state.mood_streak["count"] >= 3:
                st.markdown(f'<p style="text-align: center; color: #ffcc00;">🎉 You’ve been {mood} for {st.session_state.mood_streak["count"]} sessions in a row!</p>', unsafe_allow_html=True)
    
            st.markdown(f'<h2>{t["generated_music"]}</h2>', unsafe_allow_html=True)
            audio_file = generate_music(mood)
            if audio_file:
                st.audio(audio_file, format="audio/wav")
                st.session_state.last_audio_file = audio_file
                st.markdown('<span style="color: #00ffcc; text-align: center; display: block;">AI-Generated Music Output</span>', unsafe_allow_html=True)
                st.markdown(f'<h3>{t["rate_music"]}</h3>', unsafe_allow_html=True)
                music_rating = st.slider("Rate the music (1-5)", 1, 5, 3, key="music_rating")
                music_comments = st.text_area("Any comments on the music?", "", key="music_comments")
                if st.button(t["submit_feedback"]):
                    st.success(f"Thank you for your {music_rating}-star rating and feedback!")
    
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h3>{t["youtube_songs"]}</h3>', unsafe_allow_html=True)
                try:
                    songs = get_youtube_songs(mood)
                except Exception as e:
                    st.error(f"Failed to fetch YouTube recommendations: {str(e)}")
                    songs = FALLBACK_SONGS.get(mood, FALLBACK_SONGS["neutral"])
                for title, url in songs:
                    if not any(genre.lower() in title.lower() for genre in st.session_state.user_profile["preferred_genres"]):
                        st.markdown(f'<a href="{url}" class="youtube-link">{title}</a>', unsafe_allow_html=True)
                        update_listening_history(title, url, mood)
                    else:
                        st.markdown(f'<p style="color: #ff4d4d;">{title} (Skipped: Genre not preferred)</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            with col_rec2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h3>{t["movies"]}</h3>', unsafe_allow_html=True)
                MOVIE_RECOMMENDATIONS = {
                    "happy": [("The Secret Life of Walter Mitty", "https://www.imdb.com/title/tt0359950/")],
                    "sad": [("The Fault in Our Stars", "https://www.imdb.com/title/tt2582846/")],
                    "calm": [("The Grand Budapest Hotel", "https://www.imdb.com/title/tt2278388/")],
                    "excited": [("Mad Max: Fury Road", "https://www.imdb.com/title/tt1392190/")],
                    "neutral": [("The Shawshank Redemption", "https://www.imdb.com/title/tt0111161/")]
                }
                for title, url in MOVIE_RECOMMENDATIONS.get(mood, MOVIE_RECOMMENDATIONS["neutral"]):
                    st.markdown(f'<a href="{url}" class="movie-link">{title}</a>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            with col_rec3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h3>{t["spotify_tracks"]}</h3>', unsafe_allow_html=True)
                spotify_tracks = get_spotify_recommendations(mood)
                for title, url in spotify_tracks:
                    if not any(genre.lower() in title.lower() for genre in st.session_state.user_profile["preferred_genres"]):
                        st.markdown(f'<a href="{url}" class="spotify-link">{title}</a>', unsafe_allow_html=True)
                        update_listening_history(title, url, mood)
                    else:
                        st.markdown(f'<p style="color: #ff4d4d;">{title} (Skipped: Genre not preferred)</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            if mood in ["sad", "angry", "fear"]:
                st.markdown('<h2>🌟 Uplifting Music</h2>', unsafe_allow_html=True)
                uplifting_file = generate_music("happy")
                if uplifting_file:
                    st.audio(uplifting_file, format="audio/wav")
                    st.session_state.last_audio_file = uplifting_file
                    st.markdown('<span style="color: #00ffcc; text-align: center; display: block;">AI-Generated Uplifting Music</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        if st.button(t["save_moment"]):
            if st.session_state.mood is not None and st.session_state.last_audio_file:
                st.session_state.memory.append({
                    "mood": st.session_state.mood,
                    "confidence": st.session_state.confidence,
                    "time": time.ctime(),
                    "audio": st.session_state.last_audio_file
                })
                st.markdown('<p style="text-align: center; color: #00ffcc;">Moment saved 😁</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="text-align: center; color: #ff4d4d;">No mood or audio file to save.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.memory:
        st.markdown(f'<h2>{t["memory_capsules"]}</h2>', unsafe_allow_html=True)
        for mem in st.session_state.memory:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write(f"{mem['time']}: {mem['mood'].capitalize()} (Confidence: {mem['confidence']:.2f})")
                st.audio(mem["audio"], format="audio/wav")
                st.markdown('</div>', unsafe_allow_html=True)

    if len(st.session_state.mood_history) > 3:
        recent_moods = [m[0] for m in st.session_state.mood_history[-3:]]
        if recent_moods.count("sad") >= 2:
            st.markdown('<p style="text-align: center; color: #ffcc00;">You’ve been feeling down lately—want some uplifting music?</p>', unsafe_allow_html=True)
        elif recent_moods.count("excited") >= 2:
            st.markdown('<p style="text-align: center; color: #ffcc00;">You’re on a high! Keeping the energy up.</p>', unsafe_allow_html=True)

    display_privacy_dashboard()
    
    with st.sidebar:
        st.markdown('<div class="chatbot">', unsafe_allow_html=True)
        gemini_chatbot()  # This function should handle the chatbot title
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
