import streamlit as st
import sounddevice as sd
import wavio
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import os
import time
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import requests
import base64
import json
from urllib.parse import urlencode

class SpotifyAPI:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expiry = 0

    def get_token(self):
        """Get Spotify API token"""
        if self.token and time.time() < self.token_expiry:
            return self.token

        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')

        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}

        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            json_result = response.json()
            self.token = json_result["access_token"]
            self.token_expiry = time.time() + json_result["expires_in"] - 60  # 60 sec buffer
            return self.token
        except Exception as e:
            st.error(f"Error getting Spotify token: {e}")
            return None

    def search_by_genre(self, genre, limit=5):
        """Search for tracks based on genre"""
        token = self.get_token()
        if not token:
            return []

        genre_map = {
            "blues": "blues",
            "classical": "classical",
            "country": "country",
            "disco": "disco",
            "hiphop": "hip-hop",
            "jazz": "jazz",
            "metal": "metal",
            "pop": "pop",
            "reggae": "reggae",
            "rock": "rock"
        }

        search_genre = genre_map.get(genre.lower(), genre.lower())

        url = "https://api.spotify.com/v1/search"
        headers = {
            "Authorization": f"Bearer {token}"
        }

        params = {
            "q": f"genre:{search_genre}",
            "type": "track",
            "limit": limit
        }

        try:
            response = requests.get(f"{url}?{urlencode(params)}", headers=headers)
            response.raise_for_status()
            tracks = response.json().get("tracks", {}).get("items", [])

            recommendations = []
            for track in tracks:
                artists = ", ".join([artist["name"] for artist in track.get("artists", [])])
                recommendations.append({
                    "name": track["name"],
                    "artist": artists,
                    "album": track.get("album", {}).get("name", ""),
                    "image_url": track.get("album", {}).get("images", [{}])[0].get("url") if track.get("album", {}).get("images") else None,
                    "preview_url": track.get("preview_url"),
                    "external_url": track.get("external_urls", {}).get("spotify")
                })

            return recommendations
        except Exception as e:
            st.error(f"Error searching Spotify: {e}")
            return []

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_custom_css(theme='light'):
    if theme == 'light':
        bg_primary = '#fbda61'
        bg_secondary = '#ff5acd'
        text_primary = 'white'
        text_secondary = 'white'
        accent_color = 'purple'
        accent_hover = 'grey'
        card_shadow = 'rgba(0, 0, 0, 0.1)'
    else:  # dark mode
        bg_primary = 'black'
        bg_secondary = 'olive'
        text_primary = 'white'
        text_secondary = 'white'
        accent_color = 'yellow'
        accent_hover = 'grey'
        card_shadow = 'rgba(255, 255, 255, 0.1)'

    return f"""
    <style>
        /* Global Styles */
        body {{
            background-color: {bg_primary};
            color: {text_primary};
            transition: all 0.3s ease;
        }}

        /* Streamlit Specific Overrides */
        .stApp {{
            background: linear-gradient(135deg, {bg_primary} 0%, {bg_secondary} 100%);
        }}

        /* Main Header */
        .main-header {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {accent_color};
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 1px 1px 2px {card_shadow};
        }}

        /* Sub Header */
        .sub-header {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {text_primary};
            margin-bottom: 0.5rem;
        }}

        /* Information Text */
        .info-text {{
            font-size: 1rem;
            color: {text_secondary};
        }}

        /* Genre Prediction */
        .genre-prediction {{
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            background: linear-gradient(45deg, {accent_color}, {accent_hover});
            color: white;
            box-shadow: 0 4px 6px {card_shadow};
        }}

        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background-color: {accent_color};
        }}

        /* Improved Card Styles */
        .stCard {{
            background-color: {bg_secondary};
            border-radius: 12px;
            box-shadow: 0 4px 6px {card_shadow};
            transition: transform 0.3s ease;
        }}

        .stCard:hover {{
            transform: scale(1.02);
        }}

        /* Improved Sidebar */
        .css-1aumxhk {{
            background: linear-gradient(135deg, {bg_secondary} 0%, {bg_primary} 100%);
        }}

        /* Text Color in Sidebar and Main Area */
        .stMarkdown, .stText {{
            color: {text_primary};
        }}

        /* Button Styles */
        .stButton > button {{
            background-color: {accent_color};
            color: white;
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background-color: {accent_hover};
        }}
    </style>
    """

def theme_toggle():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    if st.sidebar.button('🌓 Toggle Theme', key = 'toggle_theme_button'):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    
    st.markdown(get_custom_css(st.session_state.theme), unsafe_allow_html=True)

def extract_features(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfccs_var = np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_feature = np.array([tempo])
    
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    
    mel_spec = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    mel_spec_var = np.var(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    return np.hstack([
        mfccs, mfccs_var, chroma, spectral_contrast, tonnetz, 
        zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff,
        tempo_feature.reshape(-1),  
        rms, mel_spec[:20], mel_spec_var[:20]  
    ])

def generate_visualizations(y, sr):
    visualizations = {}
    
    plt.figure(figsize=(10, 4), facecolor='white')
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#3F51B5')  
    plt.title('Waveform', fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)  
    plt.tight_layout()
    
    waveform = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(waveform.name, facecolor='white', edgecolor='none')
    plt.close()
    visualizations['waveform'] = waveform.name
    
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    mel_spec = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(mel_spec.name)
    plt.close()
    visualizations['mel_spectrogram'] = mel_spec.name
    
    plt.figure(figsize=(10, 4))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    
    chromagram = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chromagram.name)
    plt.close()
    visualizations['chromagram'] = chromagram.name
    
    return visualizations

@st.cache_resource
def load_models():
    models = {}
    
    try:
        models["best_ensemble"] = load("best_ensemble_genre_classifier.joblib")
        models["scaler"] = load("feature_scaler.joblib")
        models["label_encoder"] = load("label_encoder.joblib")
        
        model_files = [f for f in os.listdir('.') if f.endswith('_genre_classifier.joblib') and not f.startswith('best')]
        for model_file in model_files:
            model_name = model_file.replace('_genre_classifier.joblib', '').replace('_', ' ').title()
            models[model_name] = load(model_file)
        
    except Exception as e:
        st.warning(f"Could not load improved models, using fallback model. Error: {e}")
        models["ensemble"] = load("ensemble_genre_classifier.joblib")
        models["label_encoder"] = load("label_encoder.joblib")
    
    return models

def get_genre_characteristics(genre):
    characteristics = {
        "blues": {
            "description": "Blues is a melancholic genre characterized by blue notes, call-and-response patterns, and specific chord progressions.",
            "instruments": "Guitar, piano, harmonica, vocals",
            "origin": "Late 19th century, Deep South of the United States",
            "tempo": "Slow to moderate",
            "color": "#1E88E5"
        },
        "classical": {
            "description": "Classical music is characterized by complex compositions with sophisticated instrumental arrangements and formal structures.",
            "instruments": "Orchestra, piano, strings, woodwinds, brass",
            "origin": "Western culture, particularly from the 17th to 19th centuries",
            "tempo": "Varies widely, from very slow (largo) to very fast (prestissimo)",
            "color": "#FFC107"
        },
        "country": {
            "description": "Country music often features ballads and dance tunes with simple forms, folk lyrics, and harmonies often accompanied by string instruments.",
            "instruments": "Guitar, fiddle, banjo, harmonica, drums",
            "origin": "1920s, Southern United States",
            "tempo": "Moderate to fast",
            "color": "#FF9800"
        },
        "disco": {
            "description": "Disco is dance music with funky baselines, orchestral sounds, and four-on-the-floor beats.",
            "instruments": "Synthesizers, drums, electric guitar, bass guitar",
            "origin": "Early 1970s, urban nightlife scene in the United States",
            "tempo": "Fast (110-140 BPM)",
            "color": "#E91E63"
        },
        "hiphop": {
            "description": "Hip hop consists of stylized rhythmic music with rapping and often features sampling, drum machines, and turntablism.",
            "instruments": "Drum machine, turntables, samplers, synthesizers",
            "origin": "1970s, Bronx, New York City",
            "tempo": "Moderate to fast",
            "color": "#9C27B0"
        },
        "jazz": {
            "description": "Jazz is characterized by swing notes, blue notes, call and response, polyrhythms, and improvisation.",
            "instruments": "Saxophone, trumpet, piano, bass, drums",
            "origin": "Late 19th and early 20th century, New Orleans, United States",
            "tempo": "Varies widely",
            "color": "#3F51B5"
        },
        "metal": {
            "description": "Metal features highly distorted guitars, emphatic rhythms, dense bass-and-drum sound, and often virtuosic musicianship.",
            "instruments": "Electric guitar, bass guitar, drums, vocals",
            "origin": "Late 1960s and early 1970s, United Kingdom and United States",
            "tempo": "Often fast, sometimes slow and heavy",
            "color": "#212121"
        },
        "pop": {
            "description": "Pop music is characterized by accessible melodies, repetitive structure, and focus on recording, production, and technology.",
            "instruments": "Keyboards, synthesizers, drums, guitar, bass",
            "origin": "1950s, United States and United Kingdom",
            "tempo": "Moderate to fast",
            "color": "#FF4081"
        },
        "reggae": {
            "description": "Reggae features a heavy rhythmic accent on the off-beat with bass as the primary instrument.",
            "instruments": "Guitar, bass, drums, keyboard, horn section",
            "origin": "Late 1960s, Jamaica",
            "tempo": "Slow to moderate",
            "color": "#4CAF50"
        },
        "rock": {
            "description": "Rock music is centered on the electric guitar, featuring loud, distorted riffs and strong beats.",
            "instruments": "Electric guitar, bass guitar, drums, keyboards",
            "origin": "1950s, United States",
            "tempo": "Moderate to fast",
            "color": "#F44336"
        }
    }
    
    if genre.lower() not in characteristics:
        return {
            "description": "Information not available for this genre.",
            "instruments": "Various",
            "origin": "Unknown",
            "tempo": "Varies",
            "color": "#607D8B"
        }
    
    return characteristics[genre.lower()]

def header_section():
    import streamlit as st
import streamlit.components.v1 as components

st.markdown("""
    <style>
        .main-header {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            color: var(--text_primary);
        }
        .custom-container {
            background-color: var(--bg_primary);
            padding: 20px;
            border-radius: 12px;
            color: var(--text_primary);
        }
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            border-spacing: 0 10px;
        }
        .custom-table th, .custom-table td {
            padding: 12px;
            text-align: left;
        }
        .custom-table tr {
            background-color: var(--bg_secondary);
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎵 Music Genre Classification</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## About")
    st.markdown(
        """
        <div class="custom-container">
            <h4>🤖 Machine Learning Models</h4>
            <table class="custom-table">
                <tr>
                    <th>Model</th>
                    <th>Description</th>
                </tr>
                <tr><td><strong>🧠 SVM</strong></td><td>Support Vector Machine for non-linear classification</td></tr>
                <tr><td><strong>🌳 Random Forest</strong></td><td>Ensemble method with multiple decision trees</td></tr>
                <tr><td><strong>🚀 XGBoost</strong></td><td>Gradient boosting with high performance</td></tr>
                <tr><td><strong>🧮 Perceptron</strong></td><td>Neural network-based linear classifier</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    components.html(
        """
        <style>
            .genre-container {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 10px;
            }
            .genre-box {
                padding: 8px 14px;
                background-color: var(--bg_primary);
                color: white;
                font-weight: bold;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
        </style>
        <h2>🎵 Supported Genres</h2>
        <div class="genre-container">
            <div class="genre-box">Blues</div>
            <div class="genre-box">Classical</div>
            <div class="genre-box">Country</div>
            <div class="genre-box">Disco</div>
            <div class="genre-box">Hip Hop</div>
            <div class="genre-box">Jazz</div>
            <div class="genre-box">Metal</div>
            <div class="genre-box">Pop</div>
            <div class="genre-box">Reggae</div>
            <div class="genre-box">Rock</div>
        </div>
        """,
        height=200
    )

with col2:
    st.markdown("## Help")
    st.info(
        """
        **Tips for best results:**
        - Use clear audio samples without background noise
        - Ensure the sample contains distinguishable musical elements
        - Longer samples (10+ seconds) provide better results
        - Try different segments of a song for more accurate classification
        
        **Settings:**
        - Sample Duration: 5-30 seconds
        - Recommended: 10-15 seconds
        """
    )

st.markdown("<hr style='border-color: var(--accent_color);'>", unsafe_allow_html=True)

def sidebar_section():
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80", use_container_width=True)
        st.markdown("## About")
        st.info(
            """
            This app uses machine learning to identify music genres from audio samples.
            
            **Supported Genres:**
            - Blues
            - Classical
            - Country
            - Disco
            - Hip Hop
            - Jazz
            - Metal
            - Pop
            - Reggae
            - Rock
            
            The model was trained on the GTZAN dataset on the following models :- SVM, Random Forest Classifier, XGBoost and Perceptron.
            """
        )
        
        st.markdown("## Settings")
        sample_duration = st.slider("Sample Duration (seconds)", 5, 30, 10, key = 'slider_1')
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 20, key = 'slider_2')
        
        st.markdown("## Help")
        with st.expander("Tips for best results"):
            st.markdown(
                """
                - Use clear audio samples without background noise
                - Ensure the sample contains distinguishable musical elements
                - Longer samples (10+ seconds) provide better results
                - Try different segments of a song for more accurate classification
                """
            )

def prediction_section(models):
    st.markdown('<div class="sub-header">Upload or Record Audio</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Upload File", "Record Microphone", "Sample Library"])
    
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None
    
    with tabs[0]:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        if uploaded_file:
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}").name
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            st.session_state.audio_file_path = temp_audio_path
            
            if st.button("🔍 Analyze Genre", key="upload_analyze"):
                return temp_audio_path
    
    with tabs[1]:
        if st.button("🎤 Start Recording (30 seconds)"):
            with st.spinner("Recording..."):
                try:
                    import sounddevice as sd
                    import wavio
                    
                    fs = 22050  
                    duration = 30  
                    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                    
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(duration/100)
                        progress_bar.progress(i + 1)
                    
                    sd.wait()
                    
                    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    wavio.write(temp_audio_path, recording, fs, sampwidth=2)
                    
                    st.success("Recording complete!")
                    st.audio(temp_audio_path, format="audio/wav")
                    st.session_state.audio_file_path = temp_audio_path
                    
                    if st.button("🔍 Analyze Genre", key="record_analyze"):
                        return temp_audio_path
                    
                except Exception as e:
                    st.error(f"Error recording audio: {e}")
                    st.warning("Microphone recording may not be supported in your browser. Please try uploading a file instead.")
    
    with tabs[2]:
        st.info("Choose a sample from our library to analyze")
        sample_dir = "sample_audio"
        
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            st.warning("Sample directory created. Please add some audio samples to the 'sample_audio' folder.")
        else:
            try:
                samples = [f for f in os.listdir(sample_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
                if samples:
                    samples.insert(0, 'None')
                    selected_sample = st.selectbox("Select a sample", samples)
                    
                    if selected_sample != 'None':
                        sample_path = os.path.join(sample_dir, selected_sample)
                        st.audio(sample_path)
                        st.session_state.audio_file_path = sample_path
                    else:
                        st.session_state.audio_file_path = None
                else:
                    st.warning("No audio samples found in the sample directory. Please add some audio files to the 'sample_audio' folder.")
            except Exception as e:
                st.error(f"Error accessing sample directory: {e}")
    
    return st.session_state.audio_file_path

def analyze_audio(file_path, models):
    if not file_path:
        return None
    
    with st.spinner("Analyzing audio..."):
        try:
            y, sr = librosa.load(file_path, duration=30)
            
            features = extract_features(y, sr)
            
            visualizations = generate_visualizations(y, sr)
            
            features_reshaped = features.reshape(1, -1)
            
            if "scaler" in models:
                features_reshaped = models["scaler"].transform(features_reshaped)
            
            if "best_ensemble" in models:
                model = models["best_ensemble"]
                predicted_probs = model.predict_proba(features_reshaped)[0]
                predicted_class = model.predict(features_reshaped)[0]
            else:
                model = models["ensemble"]
                predicted_probs = model.predict_proba(features_reshaped)[0]
                predicted_class = model.predict(features_reshaped)[0]
            
            label_encoder = models["label_encoder"]
            predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
            
            top_indices = predicted_probs.argsort()[-3:][::-1]
            top_genres = label_encoder.inverse_transform(top_indices)
            top_probs = predicted_probs[top_indices]
            
            model_predictions = {}
            for name, model in models.items():
                if name not in ["best_ensemble", "ensemble", "scaler", "label_encoder"]:
                    try:
                        model_pred = model.predict(features_reshaped)[0]
                        model_genre = label_encoder.inverse_transform([model_pred])[0]
                        model_predictions[name] = model_genre
                    except Exception:
                        pass
            
            return {
                "genre": predicted_genre,
                "confidence": top_probs[0] * 100,
                "top_genres": top_genres,
                "top_probs": top_probs * 100,
                "visualizations": visualizations,
                "model_predictions": model_predictions,
                "audio_length": len(y) / sr,
                "sample_rate": sr
            }
            
        except Exception as e:
            st.error(f"Error analyzing audio: {e}")
            return None

def display_results(results):
    if not results:
        return

    genre = results["genre"]
    confidence = results["confidence"]
    characteristics = get_genre_characteristics(genre)

    st.markdown(f"""
    <div style="background-color: {characteristics['color']}; color: white; text-align: center; padding: 10px; border-radius: 10px;">
        <h2>{genre.upper()} ({confidence:.1f}%)</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        viz_tabs = st.tabs(["Waveform", "Mel Spectrogram", "Chromagram"])
        with viz_tabs[0]:
            st.image(results["visualizations"]["waveform"])
        with viz_tabs[1]:
            st.image(results["visualizations"]["mel_spectrogram"])
        with viz_tabs[2]:
            st.image(results["visualizations"]["chromagram"])

    with col2:
        st.markdown(f"### {genre.title()} Characteristics")
        st.markdown(f"**Description**: {characteristics['description']}")
        st.markdown(f"**Instruments**: {characteristics['instruments']}")
        st.markdown(f"**Origin**: {characteristics['origin']}")
        st.markdown(f"**Typical Tempo**: {characteristics['tempo']}")
        st.markdown("### Audio Statistics")
        st.markdown(f"**Duration**: {results['audio_length']:.2f} seconds")
        st.markdown(f"**Sample Rate**: {results['sample_rate']} Hz")

    st.markdown("### Confidence Levels")
    fig = go.Figure(data=[go.Pie(
        labels=results["top_genres"], 
        values=results["top_probs"], 
        hole=.5,  
        textinfo='label+percent',
        hoverinfo='label+value',
        marker=dict(colors=px.colors.qualitative.Pastel)  
    )])
    fig.update_layout(
        title="Top Genre Predictions", 
        plot_bgcolor='white',  
        width=600  
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Spotify Recommendations")
    spotify_api = SpotifyAPI(st.secrets["spotify"]["client_id"], 
                         st.secrets["spotify"]["client_secret"])
    recommended_tracks = spotify_api.search_by_genre(genre)

    if recommended_tracks:
        recommendation_cols = st.columns(len(recommended_tracks))
        for i, track in enumerate(recommended_tracks):
            with recommendation_cols[i]:
                st.markdown(f"#### [{track['name']}]({track['external_url']})")
                st.markdown(f"*{track['artist']}*")
                if track["image_url"]:
                    st.image(track["image_url"], use_container_width=True)
                if track["preview_url"]:
                    st.audio(track["preview_url"], format="audio/mp3")
    else:
        st.warning("No recommendations found for this genre.")

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    theme_toggle()
    
    header_section()
    
    models = load_models()
    
    file_path = prediction_section(models)
    
    if st.session_state.audio_file_path:
        results = analyze_audio(st.session_state.audio_file_path, models)
        
        if results:
            display_results(results)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center">
        <p>Developed by Kewal Thacker and Amiya Subhadarshi</p>
        <p>GTZAN Dataset | Ensemble Learning | Audio Processing</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
