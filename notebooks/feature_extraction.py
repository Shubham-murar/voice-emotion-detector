import librosa
import numpy as np

def extract_features(file_path):
    try:
        # Load the audio file at 16kHz
        audio, sample_rate = librosa.load(file_path, sr=16000)

        # Trim silence from beginning and end
        audio, _ = librosa.effects.trim(audio)

        # ========== 1. MFCC + Delta + Delta-Delta ==========
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_combined = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(mfcc_delta.T, axis=0),
            np.mean(mfcc_delta2.T, axis=0)
        ])  # 40 + 40 + 40 = 120 features

        # ========== 2. Zero Crossing Rate ==========
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr)

        # ========== 3. RMS (Energy) ==========
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)

        # ========== 4. Spectral Centroid ==========
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        centroid_mean = np.mean(spec_centroid)

        # ========== 5. Spectral Bandwidth ==========
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        bandwidth_mean = np.mean(spec_bw)

        # ========== 6. Chroma Features ==========
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)  # 12 features

        # Combine all features
        features = np.hstack([
            mfcc_combined,         # 120
            zcr_mean,              # 1
            rms_mean,              # 1
            centroid_mean,         # 1
            bandwidth_mean,        # 1
            chroma_mean            # 12
        ])  # Total: 136 features

        return features

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None
