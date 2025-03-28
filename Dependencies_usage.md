## 🛠️ Python Libraries & External Services

### Python Libraries:
- **streamlit**: For the web-based UI.
- **numpy**: For numerical operations (e.g., image processing, audio generation).
- **opencv-python (cv2)**: For image processing and face detection preprocessing.
- **tensorflow==2.12.0**: For loading and running the Mini-Xception model.
- **transformers**: For sentiment analysis with DistilBERT.
- **audiocraft**: For music generation with MusicGen.
- **soundfile**: For saving generated audio files.
- **google-generativeai**: For text mood analysis and chatbot via Gemini API.
- **google-api-python-client**: For YouTube API integration.
- **pandas**: For data manipulation (e.g., mood history).
- **scikit-learn**: For MinMaxScaler in biometric analysis.
- **pillow (PIL)**: For image handling.
- **torch**: Required by Audiocraft/MusicGen.
- **scipy**: For fallback audio generation.
- **matplotlib**: For mood trend visualization.
- **spotipy**: For Spotify API integration.
- **mediapipe**: For face detection in images.
- **requests**: For API calls (e.g., smartwatch, IP geolocation).
- **backoff**: For retrying failed API requests.

### External Models:
- **fer2013_mini_XCEPTION.102-0.66.hdf5**: Pre-trained Mini-Xception model for emotion detection (must be downloaded separately).
- **distilbert-base-uncased-finetuned-sst-2-english**: Pre-trained sentiment analysis model (downloaded by Transformers).
- **facebook/musicgen-small**: Pre-trained MusicGen model (downloaded by Audiocraft).

### External APIs/Services:
- **Gemini API**: For text mood analysis and chatbot (**GOOGLE_API_KEY**).
- **YouTube API**: For song recommendations (**YOUTUBE_API_KEY**).
- **Spotify API**: For track recommendations (**SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET**).
- **IP Geolocation**: Uses **ipapi.co** (no key required).
- **Smartwatch API**: Mocked, but real integration would require an API key (e.g., Fitbit API).

