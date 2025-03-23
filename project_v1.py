import os
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from concurrent.futures import ThreadPoolExecutor
from joblib import dump, load
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Audio Data Augmentation ----
def time_stretch(y, rate=0.8):
    try:
        return librosa.effects.time_stretch(y=y, rate=rate)
    except Exception as e:
        print(f"Time stretch error: {e}")
        return y

def pitch_shift(y, sr, n_steps=2):
    try:
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return y

def add_noise(y, noise_factor=0.005):
    try:
        noise = np.random.randn(len(y))
        return y + noise_factor * noise
    except Exception as e:
        print(f"Add noise error: {e}")
        return y

def time_shift(y, shift_factor=0.2):
    try:
        shift = int(len(y) * shift_factor)
        return np.roll(y, shift)
    except Exception as e:
        print(f"Time shift error: {e}")
        return y

# ---- Feature Extraction ----
def extract_features(y, sr):
    try:
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
        tempo_feature = np.array([tempo])  # This is a 1D array with a single value
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        
        # Mel spectrogram
        mel_spec = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        mel_spec_var = np.var(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        # Stack all features
        # The error occurs because tempo_feature is 1D while other arrays are also 1D but different shape
        # We need to ensure all arrays are 1D and properly shaped before stacking
        return np.hstack([
            mfccs, mfccs_var, chroma, spectral_contrast, tonnetz, 
            zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff,
            tempo_feature.reshape(-1),  # Ensure tempo_feature is properly shaped
            rms, mel_spec[:20], mel_spec_var[:20]  # Limit mel features to prevent too high dimensionality
        ])
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# ---- Process Each File with Augmentation ----
def extract_features_with_augmentation(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        if y is None or len(y) == 0:
            print(f"Skipping empty file: {file_path}")
            return []

        features = [extract_features(y, sr)]
        
        # More varied augmentation
        features.append(extract_features(time_stretch(y, 0.8), sr))
        features.append(extract_features(time_stretch(y, 1.2), sr))
        features.append(extract_features(pitch_shift(y, sr, 2), sr))
        features.append(extract_features(pitch_shift(y, sr, -2), sr))
        features.append(extract_features(add_noise(y, 0.005), sr))
        features.append(extract_features(time_shift(y, 0.2), sr))
        
        return [f for f in features if f is not None]  # Filter out failed extractions

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def process_file(file_path, label):
    feats = extract_features_with_augmentation(file_path)
    return feats, label

# ---- Load Dataset ----
def load_dataset(dataset_path):
    start_time = time.time()
    print(f"Loading dataset from {dataset_path}...")
    
    filepaths = []
    labels = []

    for folder in os.listdir(dataset_path):
        genre = folder.split("_")[0]
        folder_path = os.path.join(dataset_path, folder)
        for filename in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, filename))
            labels.append(genre)

    print(f'Total files: {len(filepaths)}')
    print(f'Dataset loaded in {time.time() - start_time:.2f} seconds')
    
    return filepaths, labels

# ---- Extract Features in Parallel ----
def extract_all_features(filepaths, labels):
    start_time = time.time()
    print("Extracting features...")
    
    features, labels_augmented = [], []
    
    max_workers = os.cpu_count() if os.cpu_count() else 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_file, filepaths, labels))

    for feats, lbl in results:
        if feats:
            features.extend(feats)
            labels_augmented.extend([lbl] * len(feats))

    print(f'Total features after augmentation: {len(features)}')
    print(f'Feature extraction completed in {time.time() - start_time:.2f} seconds')
    
    return features, labels_augmented

# ---- Main Function ----
def main():
    start_time = time.time()
    
    # Load dataset
    dataset_path = "C:\\Users\\Kewal\\Desktop\\GTZAN\\Data\\genres_original"
    filepaths, labels = load_dataset(dataset_path)
    
    # Extract features
    features, labels_augmented = extract_all_features(filepaths, labels)
    
    # Convert data for model training
    X = np.array(features)
    y = np.array(labels_augmented)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print(f'Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}')
    
    # Define and train models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # SVM
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, subsample=0.8, colsample_bytree=0.8, 
                      use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', 
                       alpha=0.0001, batch_size=64, max_iter=300, random_state=42)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    
    print('Training models...')
    models = {
        'SVM': svm,
        'Random Forest': rf,
        'XGBoost': xgb,
        'MLP': mlp,
        'Gradient Boosting': gb
    }
    
    # Train individual models
    model_accuracies = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model_start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - model_start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Training time: {train_time:.2f} seconds")
    
    # Create Voting Ensemble
    voting_ensemble = VotingClassifier(estimators=[
        ('svm', svm.best_estimator_ if hasattr(svm, 'best_estimator_') else svm),
        ('rf', rf),
        ('xgb', xgb),
        ('mlp', mlp),
        ('gb', gb)
    ], voting='soft')
    
    # Create Stacking Ensemble
    stacking_ensemble = StackingClassifier(
        estimators=[
            ('svm', svm.best_estimator_ if hasattr(svm, 'best_estimator_') else svm),
            ('rf', rf),
            ('xgb', xgb),
            ('gb', gb)
        ],
        final_estimator=mlp,
        cv=cv
    )
    
    # Train ensembles
    print("Training Voting Ensemble...")
    voting_ensemble.fit(X_train, y_train)
    
    print("Training Stacking Ensemble...")
    stacking_ensemble.fit(X_train, y_train)
    
    # Evaluate ensembles
    voting_accuracy = accuracy_score(y_test, voting_ensemble.predict(X_test))
    stacking_accuracy = accuracy_score(y_test, stacking_ensemble.predict(X_test))
    
    print(f"Voting Ensemble Accuracy: {voting_accuracy:.4f}")
    print(f"Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")
    
    # Choose the best ensemble
    best_ensemble = voting_ensemble if voting_accuracy >= stacking_accuracy else stacking_ensemble
    best_ensemble_name = "Voting Ensemble" if voting_accuracy >= stacking_accuracy else "Stacking Ensemble"
    best_accuracy = max(voting_accuracy, stacking_accuracy)
    
    print(f"Best Ensemble: {best_ensemble_name} with accuracy: {best_accuracy:.4f}")
    
    # Save best models and preprocessing objects
    dump(best_ensemble, "best_ensemble_genre_classifier.joblib")
    dump(scaler, "feature_scaler.joblib")
    dump(label_encoder, "label_encoder.joblib")
    
    # Save individual models
    for name, model in models.items():
        dump(model, f"{name.lower().replace(' ', '_')}_genre_classifier.joblib")
    
    print("All models and preprocessing objects saved successfully!")
    
    # Evaluate performance with detailed metrics
    y_pred = best_ensemble.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_, cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Save performance metrics to CSV
    performance_df = pd.DataFrame({
        'Model': list(model_accuracies.keys()) + [best_ensemble_name],
        'Accuracy': list(model_accuracies.values()) + [best_accuracy]
    })
    performance_df.to_csv('model_performance.csv', index=False)
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()