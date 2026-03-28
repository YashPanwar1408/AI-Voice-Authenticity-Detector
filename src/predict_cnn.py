"""
predict_cnn.py
--------------
Prediction module using the trained CNN model (models/cnn_model.keras).

Usage:
    from src.predict_cnn import predict_audio
    label, confidence = predict_audio("path/to/audio.wav")
"""

from __future__ import annotations

import os
import io
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore

from PIL import Image


# ── Config (must match training settings) ──────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

MODEL_CANDIDATES = (
    os.path.join(_PROJECT_ROOT, "models", "cnn_model.keras"),
    os.path.join(_PROJECT_ROOT, "models", "cnn_model.h5"),
)

MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), MODEL_CANDIDATES[0])
SAMPLE_RATE = 16000
IMG_SIZE    = 128
N_MELS      = 128
# ───────────────────────────────────────────────────────────────────────────────

# Load model once at module level
_model = None


def _load_audio_mono(file_path: str) -> tuple[np.ndarray, int]:
    """Load audio without downmix cancellation and resample to SAMPLE_RATE."""
    y, sr = librosa.load(file_path, sr=None, mono=False)

    # librosa returns (n_channels, n_samples) when mono=False
    if y.ndim == 2 and y.shape[0] > 1:
        channel_rms = np.sqrt(np.mean(np.square(y), axis=1))
        best_channel = int(np.argmax(channel_rms))
        y = y[best_channel]

    if int(sr) != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # The model was trained mostly on longer clips; very short clips often
    # get misclassified. Loop/pad short audio to a minimum duration.
    min_seconds = 5.0
    target_len = int(min_seconds * SAMPLE_RATE)
    if len(y) > 0 and len(y) < target_len:
        reps = int(np.ceil(target_len / len(y)))
        y = np.tile(y, reps)[:target_len]

    return y, int(sr)


def _get_model() -> tf.keras.Model:
    """Load and cache the CNN model."""
    global _model
    if tf is None:
        raise ImportError(
            "TensorFlow is required to run the CNN model. Install with: pip install tensorflow"
        )
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _audio_to_spectrogram(file_path: str) -> np.ndarray:
    """
    Load a .wav file and convert it to a normalised 128×128 RGB-like array.

    Returns:
        numpy array of shape (128, 128, 3), float32, values in [0, 1].
    """
    # Load audio (robust to multi-channel recordings)
    audio, sr = _load_audio_mono(file_path)

    # Mel spectrogram → dB scale
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Render *exactly like* src.generate_spectrograms.save_spectrogram()
    dpi = 100
    fig_size = IMG_SIZE / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    png_buf.seek(0)
    img_pil = Image.open(png_buf).convert("RGB")
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)

    img_arr = np.array(img_pil, dtype=np.float32)
    return img_arr / 255.0


def predict_audio(file_path: str) -> tuple[str, float]:
    """
    Predict whether an audio file is REAL or FAKE.

    Args:
        file_path: Path to the .wav audio file.

    Returns:
        label:      "REAL" or "FAKE"
        confidence: Probability score in [0.0, 1.0] for the predicted class.

    Raises:
        FileNotFoundError: If the audio file or model is missing.
        Exception:         Re-raises unexpected errors after printing them.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        model = _get_model()

        # Build spectrogram image  →  (128, 128, 3)
        img = _audio_to_spectrogram(file_path)

        # Add batch dimension  →  (1, 128, 128, 3)
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        raw_score = float(model.predict(img_batch, verbose=0)[0][0])

        # ImageDataGenerator maps alphabetically: FAKE=0, REAL=1
        # model output close to 1.0  → REAL
        # model output close to 0.0  → FAKE
        if raw_score > 0.5:
            label      = "REAL"
            confidence = raw_score
        else:
            label      = "FAKE"
            confidence = 1.0 - raw_score

        return label, round(confidence, 4)

    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"[ERROR] Prediction failed for '{file_path}': {e}")
        raise


# ── Quick self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_cnn <path_to_wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    label, confidence = predict_audio(audio_file)

    print(f"\n  File       : {audio_file}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.4f}  ({confidence * 100:.2f}%)")
