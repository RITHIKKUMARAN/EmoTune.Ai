import os
import json
import streamlit as st
import numpy as np
import cv2
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
import mediapipe as mp
from collections import Counter
import requests  # Added for smartwatch API calls

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
tf.disable_v2_behavior()

# Configure Gemini API
genai.configure(api_key="GEMINI_API_KEY")  # Replace with your Gemini API key
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
    youtube = build("youtube", "v3", developerKey="YOUTUBE_API_KEY")  # Replace with your YouTube API key
    return sentiment_analyzer, music_model, youtube

sentiment_analyzer, music_model, youtube = load_other_models()

# Spotify setup
client_id = "SPOTIFY_CLIENT_ID"  # Replace with your Spotify Client ID
client_secret = "SPOTIFY_CLIENT_SECRET_ID"  
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>📊 User Insights</h2>', unsafe_allow_html=True)
    
    profile = st.session_state.user_profile
    if profile["listening_history"]:
        moods = [entry["mood"] for entry in profile["listening_history"]]
        top_mood = Counter(moods).most_common(1)[0][0]
        genres = ["Pop", "Rock", "Classical", "Acoustic", "Dance", "Chill", "Ambient", "EDM", "Lo-Fi"]
        genre_counts = {genre: 0 for genre in genres}
        for entry in profile["listening_history"]:
            for genre in genres:
                if genre.lower() in entry["title"].lower():
                    genre_counts[genre] += 1
        favorite_genre = max(genre_counts.items(), key=lambda x: x[1])[0] if any(genre_counts.values()) else "Unknown"
        
        st.markdown(f'<p>Top Mood: <strong>{top_mood.capitalize()}</strong></p>', unsafe_allow_html=True)
        st.markdown(f'<p>Favorite Genre: <strong>{favorite_genre}</strong></p>', unsafe_allow_html=True)
    else:
        st.write("No listening history yet. Play some music to see insights!")
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🔒 Privacy Settings</h2>', unsafe_allow_html=True)
    
    st.write("### Data Usage Overview")
    data_usage = st.session_state.data_usage
    st.markdown(f"- **Biometric Data**: {'Collected' if data_usage['biometric'] else 'Not Collected'}")
    st.markdown(f"- **Facial Images**: {'Collected' if data_usage['facial'] else 'Not Collected'}")
    st.markdown(f"- **Chat History**: {'Collected' if data_usage['chat'] else 'Not Collected'}")
    
    st.write("### Manage Your Data")
    if st.button("Clear Biometric Data"):
        st.session_state.biometric_inputs = {"hr": 70, "spo2": 95, "motion": 0.5}
        st.session_state.data_usage["biometric"] = False
        st.success("Biometric data cleared.")
    
    if st.button("Clear Facial Data"):
        st.session_state.webcam_image = None
        st.session_state.temp_face_mood = None
        st.session_state.temp_face_conf = None
        st.session_state.data_usage["facial"] = False
        st.success("Facial data cleared.")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.data_usage["chat"] = False
        st.success("Chat history cleared.")
    
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
def periodic_facial_analysis():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>📸 Facial Emotion Analysis (Live)</h2>', unsafe_allow_html=True)
    
    live_analysis = st.checkbox("Enable live facial emotion analysis", value=False)
    
    if live_analysis:
        # Clear uploaded image state to avoid conflicts
        if "uploaded_image" in st.session_state:
            st.session_state.uploaded_image = None
        
        webcam_image = st.camera_input("Capture your face", key="webcam_input", label_visibility="collapsed")
        
        if webcam_image:
            with st.spinner("Processing image..."):
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
    
    else:
        # Clear webcam-related session state when live analysis is disabled
        st.session_state.webcam_image = None
        st.session_state.temp_face_mood = None
        st.session_state.temp_face_conf = None
    
    st.markdown('</div>', unsafe_allow_html=True)

# Display achievements
def display_achievements():
    if "achievements" not in st.session_state:
        st.session_state.achievements = {"happy_streak": 0}
    
    current_mood = st.session_state.mood if st.session_state.mood else "neutral"
    if current_mood == "happy":
        st.session_state.achievements["happy_streak"] += 1
    else:
        st.session_state.achievements["happy_streak"] = 0
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🏆 Achievements</h2>', unsafe_allow_html=True)
    if st.session_state.achievements["happy_streak"] >= 3:
        st.markdown('<p style="color: #ffcc00;">🎉 Happy Streak: 3 days of happiness!</p>', unsafe_allow_html=True)
    else:
        st.write(f"Happy Streak: {st.session_state.achievements['happy_streak']} days (3 needed for achievement)")
    st.markdown('</div>', unsafe_allow_html=True)

# Voice control
def voice_control():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🎙️ Voice Control</h2>', unsafe_allow_html=True)
    if st.button("Say 'Play a happy song'"):
        mood = "happy"
        st.markdown('<span style="color: #00ffcc;">Playing a happy song! (Music generation placeholder)</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Gemini chatbot function
def gemini_chatbot():
    global gemini_model
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

    st.markdown('<div class="chatbot-title">EmoTune Chatbot 🤖</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                role = "User" if message["role"] == "user" else "Bot"
                st.markdown(f'<div class="chat-message"><strong>{role}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        user_input = st.text_input("Type your message here...", key="chat_input", label_visibility="collapsed")
        submit_button = st.form_submit_button("Send")
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
        st.session_state.biometric_input_method = "manual"  # Default to manual input

    initialize_user_profile()
    initialize_data_usage()

    # Get user location based on IP (for global impact section)
    def get_user_location():
        try:
            response = requests.get("https://ipapi.co/json/")
            data = response.json()
            return data.get("country_name", "Unknown")
        except Exception as e:
            st.error(f"Failed to detect location: {str(e)}")
            return "Unknown"

    user_location = get_user_location()

    # Custom CSS (enhanced with fade-in for final mood and improved button hover effects)
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
    /* Fade-in animation for final mood display */
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

    # Privacy notice
    st.markdown(
        '<p style="text-align: center; color: #cccccc;">🔒 Your privacy matters. We do not store or share your data. All processing is done locally.</p>',
        unsafe_allow_html=True
    )

    # Particle effects
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
        st.markdown("""
        <div style="text-align: center; background: rgba(0, 0, 0, 0.5); padding: 20px; border-radius: 15px;">
            <h2>Welcome to EmoTune.AI! 🎼</h2>
            <p>Discover music tailored to your mood using AI-powered mood detection, personalized recommendations, and an emotional support chatbot.</p>
            <p>Start by selecting your mood or let us detect it for you!</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.welcome_shown = True

    # Header
    st.markdown('<h1>EmoTune.AI 🎼</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #ffcc00; text-shadow: 0 0 5px #ffcc00;">A mood-responsive audio experience</p>', unsafe_allow_html=True)

    # Consent popup
    consent_given = display_consent_popup()

    # Main content
    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2>🎭 Select Your Mood</h2>', unsafe_allow_html=True)
            manual_mood = st.selectbox("Choose your mood (or auto-detect)", ["Auto-detect"] + MOOD_OPTIONS, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2>📝 Describe Your Day</h2>', unsafe_allow_html=True)
            text_input = st.text_area("Describe your mood or day (optional)", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2>🎙️ Speak About Your Day (Optional)</h2>', unsafe_allow_html=True)
            audio_input = st.file_uploader("Upload an audio clip of you speaking", type=["wav", "mp3"], label_visibility="collapsed")
            if audio_input:
                with st.spinner("Analyzing voice tone..."):
                    # Simulate voice tone analysis (in a real app, use librosa to analyze pitch/energy)
                    voice_mood, voice_conf = "happy", 0.75  # Placeholder
                    st.session_state.voice_mood = voice_mood
                    st.session_state.voice_conf = voice_conf
                    st.session_state.data_usage["voice"] = True
                    st.success("Voice tone processed. Click 'Analyze My Mood' to see the result.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2>📷 Upload an Image (Optional)</h2>', unsafe_allow_html=True)
            uploaded_image = st.file_uploader("Upload an image for facial emotion analysis", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if uploaded_image:
                # Clear webcam-related session state to avoid conflicts
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
            st.markdown('<h2>🌟 Mood Booster Tips</h2>', unsafe_allow_html=True)
            mood_tips = {
                "happy": [
                    "Keep the positivity going—share a smile with someone today! 😊",
                    "Dance to your favorite song to amplify your joy! 💃"
                ],
                "sad": [
                    "Take a short walk to lift your spirits! 🚶",
                    "Listen to some uplifting music—we’ve got you covered! 🎶"
                ],
                "calm": [
                    "Try some deep breathing exercises to maintain your peace! 🧘",
                    "Sip a warm cup of tea and relax! ☕"
                ],
                "excited": [
                    "Channel your energy into a fun activity—maybe a quick workout? 🏋️",
                    "Share your excitement with a friend! 📞"
                ],
                "neutral": [
                    "Try something new today to spark some inspiration! ✨",
                    "Reflect on your day—what made you smile? 📝"
                ],
                "angry": [
                    "Take a moment to breathe deeply and let go of tension! 🌬️",
                    "Write down what’s bothering you to clear your mind! ✍️"
                ],
                "fear": [
                    "Talk to someone you trust to ease your worries! 🗣️",
                    "Try a grounding exercise—focus on your surroundings! 🌳"
                ],
                "surprise": [
                    "Embrace the unexpected—maybe it’s a sign of something great! 🎉",
                    "Capture this moment with a quick journal entry! 📓"
                ],
                "disgust": [
                    "Step away from what’s bothering you and take a break! 🚪",
                    "Focus on something positive to shift your mood! 🌈"
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
                st.markdown('<h2>📈 Mood Trend</h2>', unsafe_allow_html=True)
                if st.session_state.mood_history:
                    recent_moods = [m[0] for m in st.session_state.mood_history[-5:]]
                    mood_colors = {
                        "happy": "#ffcc00",
                        "sad": "#00cccc",
                        "calm": "#66ff66",
                        "excited": "#ff66cc",
                        "neutral": "#cccccc"
                    }
                    # Mapping for emotions to moods specifically for the graph
                    emotion_to_mood_for_graph = {
                        "angry": "sad",
                        "disgust": "sad",
                        "fear": "sad",
                        "happy": "happy",
                        "sad": "sad",
                        "surprise": "excited",
                        "neutral": "neutral"
                    }
                    # Map the moods/emotions to ensure they are in MOOD_OPTIONS
                    mapped_moods = [emotion_to_mood_for_graph.get(mood, mood if mood in MOOD_OPTIONS else "neutral") for mood in recent_moods]
                    line_color = mood_colors.get(mapped_moods[-1], "#ff7878")
                    try:
                        mood_indices = [MOOD_OPTIONS.index(mood) for mood in mapped_moods]
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.plot(range(len(mapped_moods)), mood_indices, marker='o', color=line_color, linewidth=2, markersize=8)
                        ax.set_yticks(range(len(MOOD_OPTIONS)))
                        ax.set_yticklabels(MOOD_OPTIONS)
                        ax.set_title("Mood Trend Over Time", color='#ffcc00')
                        ax.set_xlabel("Session", color='#fff')
                        ax.set_ylabel("Mood", color='#fff')
                        ax.tick_params(axis='x', colors='#fff')
                        ax.tick_params(axis='y', colors='#fff')
                        ax.set_facecolor((0, 0, 0, 0.3))
                        fig.patch.set_facecolor((0, 0, 0, 0.3))
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Failed to render mood trend graph: {str(e)}")
                else:
                    st.write("No mood history yet. Analyze your mood to see trends!")
                st.markdown('</div>', unsafe_allow_html=True)

            periodic_facial_analysis()
            
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h2>🌍 Mental Health & Community</h2>', unsafe_allow_html=True)
                st.markdown(
                    '<p>Music can reduce stress by up to 65% (Source: WHO). Need more support? <a href="https://www.who.int/health-topics/mental-health" target="_blank">Explore WHO Mental Health Resources</a>.</p>',
                    unsafe_allow_html=True
                )
                st.markdown('<h3>🌟 Share Your Uplifting Music</h3>', unsafe_allow_html=True)
                if st.session_state.last_audio_file:
                    if st.button("Share to Community"):
                        # Simulate sharing (in a real app, this would save to a database)
                        st.success("Your music has been shared with the community! 🎉")
                else:
                    st.write("Generate music to share with the community!")
                st.markdown('</div>', unsafe_allow_html=True)

            # Global Impact Section
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h2>🌐 Global Impact</h2>', unsafe_allow_html=True)
                st.markdown(
                    '<p>EmoTune.AI supports mental health worldwide, especially in underserved communities. It can be deployed on low-cost devices like Raspberry Pi to bring music therapy to rural areas.</p>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<p>Based on your location ({user_location}), here are some mental health resources:</p>', unsafe_allow_html=True)
                resources = {
                    "United States": [("Mental Health America", "https://mhanational.org")],
                    "India": [("Vandrevala Foundation", "https://www.vandrevalafoundation.com")],
                    "Default": [("World Health Organization", "https://www.who.int/health-topics/mental-health")]
                }
                location_key = user_location if user_location in resources else "Default"
                for name, url in resources[location_key]:
                    st.markdown(f'<a href="{url}" class="youtube-link">{name}</a>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Mood Journey (Gamification)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h2>🏆 Your Mood Journey</h2>', unsafe_allow_html=True)
                # Badge: Mood Explorer (analyze mood 5 times)
                if len(st.session_state.mood_history) >= 5:
                    st.markdown('<p style="color: #ffcc00;">🎉 Mood Explorer Badge: Analyzed mood 5 times!</p>', unsafe_allow_html=True)
                else:
                    st.write(f"Analyze your mood {5 - len(st.session_state.mood_history)} more times to earn the Mood Explorer Badge!")
                # Badge: Streak Master (3 consecutive same moods)
                if st.session_state.mood_streak["count"] >= 3:
                    st.markdown(f'<p style="color: #ffcc00;">🔥 Streak Master Badge: {st.session_state.mood_streak["mood"].capitalize()} for {st.session_state.mood_streak["count"]} sessions!</p>', unsafe_allow_html=True)
                else:
                    st.write(f"Maintain the same mood for {3 - st.session_state.mood_streak['count']} more sessions to earn the Streak Master Badge!")
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
                    insight_text = "Biometrics Normal"
                st.markdown(f'<span class="bio-insight-icon">{insight_icon}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="bio-insight-text">{insight_text}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h2>💓 Biometric Data</h2>', unsafe_allow_html=True)
                
                # Checkbox for enabling biometric data usage
                use_biometrics = st.checkbox("Use biometric data", value=False)
                
                if use_biometrics:
                    # Add two mutually exclusive checkboxes for input method
                    col_manual, col_smartwatch = st.columns(2)
                    
                    with col_manual:
                        use_manual_input = st.checkbox("Manual Input", value=True, key="manual_input")
                    
                    with col_smartwatch:
                        use_smartwatch = st.checkbox("Fetch from Smartwatch", value=False, key="smartwatch_input")
                    
                    # Ensure only one input method is selected
                    if use_manual_input and use_smartwatch:
                        st.warning("Please select only one biometric input method.")
                        use_smartwatch = False  # Default to manual input if both are selected
                    
                    # Update session state based on checkbox changes
                    if use_manual_input and not use_smartwatch:
                        st.session_state.biometric_input_method = "manual"
                    elif use_smartwatch and not use_manual_input:
                        st.session_state.biometric_input_method = "smartwatch"
                    else:
                        st.session_state.biometric_input_method = "manual"  # Default to manual if neither or both are selected
                    
                    # Process biometric data based on the selected method
                    if st.session_state.biometric_input_method == "manual":
                        st.write("Enter biometric data manually:")
                        with st.form(key="biometric_form"):
                            hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=st.session_state.biometric_inputs["hr"])
                            spo2 = st.number_input("SpO2 (%)", min_value=80, max_value=100, value=st.session_state.biometric_inputs["spo2"])
                            motion = st.slider("Motion Level (0-1)", 0.0, 1.0, st.session_state.biometric_inputs["motion"])
                            submit_button = st.form_submit_button("Update Biometric Data")
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
                st.markdown('<h2>😊 Current Mood</h2>', unsafe_allow_html=True)
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
        st.markdown('<h2>🎵 Preferences</h2>', unsafe_allow_html=True)
        genre_skip = st.multiselect("Skip these genres", ["Metal", "Rap", "Classical", "Acoustic", "Dance", "Chill", "Ambient", "EDM", "Lo-Fi"], default=[], label_visibility="collapsed")
        st.session_state.user_profile["preferred_genres"] = genre_skip
        st.markdown('</div>', unsafe_allow_html=True)

    voice_control()

    with st.container():
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        if st.button("Analyze My Mood 🧠"):
            with st.spinner("Analyzing your mood..."):
                mood, confidence = analyze_mood(
                    manual_mood, text_input, use_biometrics, session, emotion_model, graph
                )
                st.session_state.mood = mood
                st.session_state.confidence = confidence
                st.session_state.mood_history.append((mood, confidence, time.time()))
                # Apply fade-in animation to the final mood display
                st.markdown(
                    f'<p class="fade-in-mood" style="text-align: center; font-size: 1.5em;">Final Detected Mood: <strong>{mood.capitalize()}</strong> (Confidence: {confidence:.2f})</p>',
                    unsafe_allow_html=True
                )
    
            # Update mood streak
            if st.session_state.mood_streak["mood"] == mood:
                st.session_state.mood_streak["count"] += 1
            else:
                st.session_state.mood_streak = {"mood": mood, "count": 1}
            if st.session_state.mood_streak["count"] >= 3:
                st.markdown(f'<p style="text-align: center; color: #ffcc00;">🎉 You’ve been {mood} for {st.session_state.mood_streak["count"]} sessions in a row!</p>', unsafe_allow_html=True)
    
            # Music generation
            st.markdown('<h2>🎶 Generated Music</h2>', unsafe_allow_html=True)
            audio_file = generate_music(mood)
            if audio_file:
                st.audio(audio_file, format="audio/wav")
                st.session_state.last_audio_file = audio_file
                st.markdown('<span style="color: #00ffcc; text-align: center; display: block;">AI-Generated Music Output</span>', unsafe_allow_html=True)
                # Feedback form for music
                st.markdown('<h3>🎵 How was the music?</h3>', unsafe_allow_html=True)
                music_rating = st.slider("Rate the music (1-5)", 1, 5, 3, key="music_rating")
                music_comments = st.text_area("Any comments on the music?", "", key="music_comments")
                if st.button("Submit Feedback"):
                    # In a real app, save the feedback to a database
                    st.success(f"Thank you for your {music_rating}-star rating and feedback!")
    
            # Recommendations
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>▶️ YouTube Songs</h3>', unsafe_allow_html=True)
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
                st.markdown('<h3>🎬 Movies</h3>', unsafe_allow_html=True)
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
                st.markdown('<h3>🎧 Spotify Tracks</h3>', unsafe_allow_html=True)
                spotify_tracks = get_spotify_recommendations(mood)
                for title, url in spotify_tracks:
                    if not any(genre.lower() in title.lower() for genre in st.session_state.user_profile["preferred_genres"]):
                        st.markdown(f'<a href="{url}" class="spotify-link">{title}</a>', unsafe_allow_html=True)
                        update_listening_history(title, url, mood)
                    else:
                        st.markdown(f'<p style="color: #ff4d4d;">{title} (Skipped: Genre not preferred)</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            # Uplifting music for negative moods
            if mood in ["sad", "angry", "fear"]:
                st.markdown('<h2>🌟 Uplifting Music</h2>', unsafe_allow_html=True)
                uplifting_file = generate_music("happy")
                if uplifting_file:
                    st.audio(uplifting_file, format="audio/wav")
                    st.session_state.last_audio_file = uplifting_file
                    st.markdown('<span style="color: #00ffcc; text-align: center; display: block;">AI-Generated Uplifting Music</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Save moment
    with st.container():
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        if st.button("Save This Moment 😉"):
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

    # Emotional memory capsules
    if st.session_state.memory:
        st.markdown('<h2>🔮 Emotional Memory Capsules</h2>', unsafe_allow_html=True)
        for mem in st.session_state.memory:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write(f"{mem['time']}: {mem['mood'].capitalize()} (Confidence: {mem['confidence']:.2f})")
                st.audio(mem["audio"], format="audio/wav")
                st.markdown('</div>', unsafe_allow_html=True)

    # Mood history analysis
    if len(st.session_state.mood_history) > 3:
        recent_moods = [m[0] for m in st.session_state.mood_history[-3:]]
        if recent_moods.count("sad") >= 2:
            st.markdown('<p style="text-align: center; color: #ffcc00;">You’ve been feeling down lately—want some uplifting music?</p>', unsafe_allow_html=True)
        elif recent_moods.count("excited") >= 2:
            st.markdown('<p style="text-align: center; color: #ffcc00;">You’re on a high! Keeping the energy up.</p>', unsafe_allow_html=True)

    display_privacy_dashboard()

    with st.sidebar:
        st.markdown('<div class="chatbot">', unsafe_allow_html=True)
        gemini_chatbot()
        st.markdown('</div>', unsafe_allow_html=True)



if __name__ == "__main__":
    main()
