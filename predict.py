import joblib
import sys
import numpy as np
from notebooks.feature_extraction import extract_features

def predict_emotion(audio_path):
    # Load model, scaler, and label encoder
    model = joblib.load("models/voice_emotion_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    # Extract features
    features = extract_features(audio_path)
    if features is None:
        print("❌ Could not extract features from audio.")
        return

    # Scale features
    features_scaled = scaler.transform([features])

    # Predict probabilities for all classes
    probabilities = model.predict_proba(features_scaled)[0]
    emotion_scores = {
        label: float(f"{prob:.2f}")
        for label, prob in zip(label_encoder.classes_, probabilities)
    }

    # Sort by confidence
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

    # Display results
    print("🎯 Emotion Prediction (sorted by confidence):")
    for emotion, score in sorted_emotions:
        bar = "█" * int(score * 20)
        print(f"{emotion.capitalize():<8}: {score:<5} {bar}")

    # Optional: Top pick
    top_emotion, top_score = sorted_emotions[0]
    print(f"\n🗣️ Most Likely Emotion: {top_emotion} (Confidence: {top_score})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("⚠️ Usage: python predict.py <path_to_audio.wav>")
    else:
        audio_path = sys.argv[1]
        predict_emotion(audio_path)
