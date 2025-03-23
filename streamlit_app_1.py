import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #191414;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555555;
    }
    .genre-prediction {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #1DB954;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

def extract_features(y, sr):
    # Basic features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfccs_var = np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    
    # Additional features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
    
    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_feature = np.array([tempo])
    
    # RMS energy
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    
    # Mel spectrogram
    mel_spec = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    mel_spec_var = np.var(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    # Stack all features
    return np.hstack([
        mfccs, mfccs_var, chroma, spectral_contrast, tonnetz, 
        zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff,
        tempo_feature.reshape(-1),  # Ensure tempo_feature is properly shaped
        rms, mel_spec[:20], mel_spec_var[:20]  # Limit mel features to prevent too high dimensionality
    ])

# Function to generate audio visualizations
def generate_visualizations(y, sr):
    # Create visualizations
    visualizations = {}
    
    # Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    
    waveform = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(waveform.name)
    plt.close()
    visualizations['waveform'] = waveform.name
    
    # Mel Spectrogram
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
    
    # Chromagram
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

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    models = {}
    
    try:
        # Try to load best ensemble model first
        models["best_ensemble"] = load("best_ensemble_genre_classifier.joblib")
        models["scaler"] = load("feature_scaler.joblib")
        models["label_encoder"] = load("label_encoder.joblib")
        
        # Try to load individual models if available
        model_files = [f for f in os.listdir('.') if f.endswith('_genre_classifier.joblib') and not f.startswith('best')]
        for model_file in model_files:
            model_name = model_file.replace('_genre_classifier.joblib', '').replace('_', ' ').title()
            models[model_name] = load(model_file)
        
    except Exception as e:
        # Fallback to original model
        st.warning(f"Could not load improved models, using fallback model. Error: {e}")
        models["ensemble"] = load("ensemble_genre_classifier.joblib")
        models["label_encoder"] = load("label_encoder.joblib")
    
    return models

# Get genre characteristics
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
    
    # Return default characteristics if genre not found
    if genre.lower() not in characteristics:
        return {
            "description": "Information not available for this genre.",
            "instruments": "Various",
            "origin": "Unknown",
            "tempo": "Varies",
            "color": "#607D8B"
        }
    
    return characteristics[genre.lower()]

# Define app sections
def header_section():
    st.markdown('<div class="main-header">🎵 Music Genre Classification</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-text">
        Upload an audio file or record a sample to identify its musical genre. 
        The app analyzes audio characteristics to determine the most likely genre.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Add a cool horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)

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
        sample_duration = st.slider("Sample Duration (seconds)", 5, 30, 10)
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 20)
        
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
    
    # Create tabs for different input methods
    tabs = st.tabs(["Upload File", "Record Microphone", "Sample Library"])
    
    audio_data = None
    file_path = None
    
    # Upload tab
    with tabs[0]:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        if uploaded_file:
            # Save uploaded file to temporary location
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}").name
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            file_path = temp_audio_path
    
    # Recording tab
    with tabs[1]:
        if st.button("🎤 Start Recording (5 seconds)"):
            with st.spinner("Recording..."):
                try:
                    import sounddevice as sd
                    import wavio
                    
                    fs = 22050  # Sample rate
                    duration = 5  # seconds
                    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                    
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(duration/100)
                        progress_bar.progress(i + 1)
                    
                    sd.wait()
                    
                    # Save the recording
                    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    wavio.write(temp_audio_path, recording, fs, sampwidth=2)
                    
                    st.success("Recording complete!")
                    st.audio(temp_audio_path, format="audio/wav")
                    file_path = temp_audio_path
                    
                except Exception as e:
                    st.error(f"Error recording audio: {e}")
                    st.warning("Microphone recording may not be supported in your browser. Please try uploading a file instead.")
    
    # Sample library tab
    with tabs[2]:
        st.info("Choose a sample from our library to analyze")
        sample_dir = "sample_audio"
        
        # Create sample directory if it doesn't exist
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            st.warning("Sample directory created. Please add some audio samples to the 'sample_audio' folder.")
        else:
            try:
                samples = [f for f in os.listdir(sample_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
                if samples:
                    selected_sample = st.selectbox("Select a sample", samples)
                    sample_path = os.path.join(sample_dir, selected_sample)
                    st.audio(sample_path)
                    file_path = sample_path
                else:
                    st.warning("No audio samples found in the sample directory. Please add some audio files to the 'sample_audio' folder.")
            except Exception as e:
                st.error(f"Error accessing sample directory: {e}")
    
    return file_path

def analyze_audio(file_path, models):
    if not file_path:
        return None
    
    with st.spinner("Analyzing audio..."):
        try:
            # Load the audio
            y, sr = librosa.load(file_path, duration=30)
            
            # Extract features
            features = extract_features(y, sr)
            
            # Generate visualizations
            visualizations = generate_visualizations(y, sr)
            
            # Prepare features for model
            features_reshaped = features.reshape(1, -1)
            
            # Scale features if scaler is available
            if "scaler" in models:
                features_reshaped = models["scaler"].transform(features_reshaped)
            
            # Make prediction with best ensemble if available, otherwise use fallback model
            if "best_ensemble" in models:
                model = models["best_ensemble"]
                predicted_probs = model.predict_proba(features_reshaped)[0]
                predicted_class = model.predict(features_reshaped)[0]
            else:
                model = models["ensemble"]
                predicted_probs = model.predict_proba(features_reshaped)[0]
                predicted_class = model.predict(features_reshaped)[0]
            
            # Get genre label
            label_encoder = models["label_encoder"]
            predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
            
            # Get top 3 predictions
            top_indices = predicted_probs.argsort()[-3:][::-1]
            top_genres = label_encoder.inverse_transform(top_indices)
            top_probs = predicted_probs[top_indices]
            
            # Get additional predictions from individual models if available
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
    
    # Display the main prediction
    genre = results["genre"]
    confidence = results["confidence"]
    
    # Get genre characteristics
    characteristics = get_genre_characteristics(genre)
    
    st.markdown(f"""
    <div class="genre-prediction" style="background-color: {characteristics['color']};">
        {genre.upper()} ({confidence:.1f}%)
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for visualization and details
    col1, col2 = st.columns([2, 1])
    
    # Column 1: Visualizations in tabs
    with col1:
        viz_tabs = st.tabs(["Waveform", "Mel Spectrogram", "Chromagram"])
        
        with viz_tabs[0]:
            st.image(results["visualizations"]["waveform"])
            st.caption("Waveform visualization shows the amplitude of the audio signal over time.")
            
        with viz_tabs[1]:
            st.image(results["visualizations"]["mel_spectrogram"])
            st.caption("Mel Spectrogram shows the frequency content of the audio over time on a mel scale.")
            
        with viz_tabs[2]:
            st.image(results["visualizations"]["chromagram"])
            st.caption("Chromagram shows the distribution of energy across the 12 pitch classes.")
    
    # Column 2: Genre details and characteristics
    with col2:
        st.markdown(f"### {genre.title()} Characteristics")
        st.markdown(f"**Description**: {characteristics['description']}")
        st.markdown(f"**Instruments**: {characteristics['instruments']}")
        st.markdown(f"**Origin**: {characteristics['origin']}")
        st.markdown(f"**Typical Tempo**: {characteristics['tempo']}")
        
        # Display audio stats
        st.markdown("### Audio Statistics")
        st.markdown(f"**Duration**: {results['audio_length']:.2f} seconds")
        st.markdown(f"**Sample Rate**: {results['sample_rate']} Hz")
    
    # Display prediction confidence for top genres
    st.markdown("### Confidence Levels")
    
    # Create a bar chart for genre confidence
    fig = px.bar(
        x=results["top_probs"],
        y=results["top_genres"],
        orientation='h',
        labels={"x": "Confidence (%)", "y": "Genre"},
        text=[f"{p:.1f}%" for p in results["top_probs"]],
        color=results["top_probs"],
        color_continuous_scale=["blue", "green", "#1DB954"],
        title="Top Genre Predictions"
    )
    
    fig.update_layout(
        xaxis_range=[0, 100],
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display individual model predictions if available
    if results["model_predictions"]:
        st.markdown("### Individual Model Predictions")
        model_df = pd.DataFrame({
            "Model": list(results["model_predictions"].keys()),
            "Predicted Genre": list(results["model_predictions"].values())
        })
        st.dataframe(model_df, use_container_width=True)
    
    # Similar artists/songs recommendation (placeholder)
    with st.expander("Similar Artists & Songs"):
        st.info("This is a placeholder for similar artists and songs recommendations. In a full implementation, this would connect to a music database or API.")
        st.markdown(f"Artists typically associated with {genre} include:")
        
        # Mock data for demonstration
        if genre.lower() == "rock":
            artists = ["Led Zeppelin", "Queen", "AC/DC", "The Rolling Stones", "Nirvana"]
        elif genre.lower() == "jazz":
            artists = ["Miles Davis", "John Coltrane", "Ella Fitzgerald", "Louis Armstrong", "Thelonious Monk"]
        elif genre.lower() == "hiphop":
            artists = ["Kendrick Lamar", "Jay-Z", "Tupac Shakur", "Nas", "The Notorious B.I.G."]
        else:
            artists = ["Artist 1", "Artist 2", "Artist 3", "Artist 4", "Artist 5"]
            
        for artist in artists:
            st.markdown(f"- {artist}")

def main():
    # Configure page
    header_section()
    sidebar_section()
    
    # Load models
    models = load_models()
    
    # Get audio input
    file_path = prediction_section(models)
    
    # Analyze button
    if file_path and st.button("🔍 Analyze Genre"):
        results = analyze_audio(file_path, models)
        
        if results:
            display_results(results)
            
            # Clean up temporary files
            for viz_file in results["visualizations"].values():
                try:
                    if os.path.exists(viz_file):
                        os.unlink(viz_file)
                except Exception:
                    pass
    
    # Footer
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