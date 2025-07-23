import os
import numpy as np
from notebooks.feature_extraction import extract_features

# Emotion map
EMOTION_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral"
}

def load_dataset(folder_path):
    features = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            parts = filename.split("_")
            emotion_code = parts[2]

            if emotion_code not in EMOTION_MAP:
                continue  # skip unwanted emotions

            emotion_label = EMOTION_MAP[emotion_code]
            file_path = os.path.join(folder_path, filename)

            try:
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(emotion_label)
            except:
                print(f"❌ Failed to process {filename}")

    return np.array(features), np.array(labels)
