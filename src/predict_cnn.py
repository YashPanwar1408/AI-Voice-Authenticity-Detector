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
import numpy as np

try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore

# Optional: use librosa + matplotlib for preprocessing when available.
# This matches the training spectrogram pipeline more closely than the
# TensorFlow-only fallback (especially colormap + STFT padding behavior).
try:  # pragma: no cover
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

try:  # pragma: no cover
    from matplotlib import cm as mpl_cm  # type: ignore
except Exception:  # pragma: no cover
    mpl_cm = None  # type: ignore

from PIL import Image

from src.gradcam import generate_gradcam, overlay_heatmap


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
N_FFT       = 2048
HOP_LENGTH  = 512
TOP_DB      = 80.0
# ───────────────────────────────────────────────────────────────────────────────

# Load model once at module level
_model = None
_mel_w = None
_mpl_magma = None


def _linear_resample_1d(x: tf.Tensor, n_out: tf.Tensor) -> tf.Tensor:
    """Linearly resample a 1D signal to `n_out` samples (TensorFlow-only).

    We avoid optional deps (librosa/tensorflow-io). This is sufficient for
    inference-time resampling when inputs aren't already at `SAMPLE_RATE`.
    """
    if tf is None:
        raise ImportError("TensorFlow is required.")

    x = tf.cast(tf.reshape(x, [-1]), tf.float32)
    n_out = tf.maximum(tf.cast(n_out, tf.int32), 1)
    n_in = tf.shape(x)[0]

    def _tile_first() -> tf.Tensor:
        first = x[:1]
        first = tf.cond(
            tf.shape(first)[0] > 0,
            lambda: first,
            lambda: tf.zeros([1], dtype=tf.float32),
        )
        return tf.tile(first, [n_out])

    def _interp() -> tf.Tensor:
        n_in_f = tf.cast(n_in, tf.float32)
        pos = tf.linspace(0.0, n_in_f - 1.0, n_out)
        idx0 = tf.cast(tf.floor(pos), tf.int32)
        idx1 = tf.minimum(idx0 + 1, n_in - 1)
        w = pos - tf.cast(idx0, tf.float32)
        x0 = tf.gather(x, idx0)
        x1 = tf.gather(x, idx1)
        return (1.0 - w) * x0 + w * x1

    return tf.cond(tf.logical_or(n_in < 2, n_out < 2), _tile_first, _interp)


def _hz_to_mel_slaney(freq_hz: np.ndarray) -> np.ndarray:
    """Hz → mel (Slaney-style), matching librosa when htk=False."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = freq_hz / f_sp

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    log_t = freq_hz >= min_log_hz
    mels[log_t] = min_log_mel + np.log(freq_hz[log_t] / min_log_hz) / logstep
    return mels.astype(np.float64)


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Mel (Slaney-style) → Hz, matching librosa when htk=False."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    freqs = f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs.astype(np.float64)


def _librosa_mel_filterbank(
    *,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Create a librosa-style mel filterbank (htk=False, norm='slaney').

    Returns:
        weights: shape (n_mels, 1 + n_fft//2), float32
    """
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, float(sr) / 2.0, n_freqs, dtype=np.float64)

    mel_min = _hz_to_mel_slaney(np.array([fmin], dtype=np.float64))[0]
    mel_max = _hz_to_mel_slaney(np.array([fmax], dtype=np.float64))[0]
    mels = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz = _mel_to_hz_slaney(mels)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left = hz[i]
        center = hz[i + 1]
        right = hz[i + 2]

        # Avoid division by zero in pathological cases
        if center <= left:
            center = left + 1e-6
        if right <= center:
            right = center + 1e-6

        lower = (fft_freqs - left) / (center - left)
        upper = (right - fft_freqs) / (right - center)
        weights[i, :] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney-style normalization: constant energy per channel
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[0:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights.astype(np.float32)


def _get_mel_weight_matrix() -> tf.Tensor:
    """Return cached mel weights shaped for tf.matmul(power_spec, W)."""
    global _mel_w
    if tf is None:
        raise ImportError("TensorFlow is required.")
    if _mel_w is None:
        mel_basis = _librosa_mel_filterbank(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=0.0,
            fmax=float(SAMPLE_RATE) / 2.0,
        )
        # librosa returns (n_mels, n_freqs); we want (n_freqs, n_mels)
        _mel_w = tf.constant(mel_basis.T, dtype=tf.float32)
    return _mel_w


def _load_audio_mono(file_path: str) -> tuple[tf.Tensor, int]:
    """Decode WAV and return mono float32 audio at SAMPLE_RATE.

    Streamlit Cloud note: we avoid librosa/soundfile and rely on TensorFlow.
    The Streamlit app converts uploads to PCM S16LE WAV at 16 kHz.
    """
    if tf is None:
        raise ImportError("TensorFlow is required.")

    wav_bytes = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=-1)
    sr_i = int(sr.numpy())

    # audio shape: (n_samples, n_channels)
    if audio.shape.rank == 2 and int(audio.shape[1]) > 1:
        audio = tf.reduce_mean(audio, axis=1)
    else:
        audio = tf.squeeze(audio, axis=-1) if audio.shape.rank == 2 else tf.squeeze(audio)

    # Resample if needed (shouldn't happen if conversion forced 16 kHz)
    if sr_i != SAMPLE_RATE and sr_i > 0:
        n_in = tf.shape(audio)[0]
        n_out = tf.cast(tf.round(tf.cast(n_in, tf.float32) * (SAMPLE_RATE / float(sr_i))), tf.int32)
        audio = _linear_resample_1d(audio, n_out)
        sr_i = SAMPLE_RATE

    # Short-clip fix: loop to minimum duration
    min_seconds = 5.0
    target_len = int(min_seconds * SAMPLE_RATE)
    n = int(audio.shape[0]) if audio.shape.rank == 1 and audio.shape[0] is not None else None
    if n is not None and n > 0 and n < target_len:
        reps = int(np.ceil(target_len / n))
        audio = tf.tile(audio, [reps])[:target_len]
    else:
        n_dyn = tf.shape(audio)[0]
        audio = tf.cond(
            tf.logical_and(n_dyn > 0, n_dyn < target_len),
            lambda: tf.tile(audio, [tf.cast(tf.math.ceil(target_len / tf.cast(n_dyn, tf.float32)), tf.int32)])[:target_len],
            lambda: audio,
        )

    return tf.cast(audio, tf.float32), sr_i


def _magma_like_colormap_256() -> np.ndarray:
    """A small magma-like ramp (not exact), 256x3 float32 in [0,1]."""
    # Anchor points (x in [0,1]) from dark purple -> red -> orange -> yellow.
    anchors_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    anchors_rgb = np.array(
        [
            [0.02, 0.01, 0.10],
            [0.25, 0.03, 0.35],
            [0.65, 0.12, 0.25],
            [0.95, 0.45, 0.10],
            [0.99, 0.95, 0.55],
        ],
        dtype=np.float32,
    )
    xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    out = np.zeros((256, 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = np.interp(xs, anchors_x, anchors_rgb[:, c]).astype(np.float32)
    return out


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


def _get_mpl_magma():
    global _mpl_magma
    if _mpl_magma is None:
        if mpl_cm is None:  # pragma: no cover
            raise ImportError("matplotlib is required for magma colormap")
        _mpl_magma = mpl_cm.get_cmap("magma")
    return _mpl_magma


def _audio_to_spectrogram_librosa(file_path: str) -> np.ndarray:
    """Training-aligned preprocessing using librosa + matplotlib.

    This mirrors `src/generate_spectrograms.py` (mel spectrogram in dB, magma
    colormap, origin='lower') without writing an intermediate PNG.

    Returns:
        numpy array of shape (128, 128, 3), float32, values in [0, 1].
    """
    if librosa is None or mpl_cm is None:
        raise ImportError("librosa/matplotlib not available")

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB).astype(np.float32)

    # Match matplotlib imshow(origin='lower')
    mel_db = np.flipud(mel_db)

    vmin = float(np.min(mel_db))
    vmax = float(np.max(mel_db))
    denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1.0
    norm = (mel_db - vmin) / denom
    norm = np.clip(norm, 0.0, 1.0)

    cmap = _get_mpl_magma()
    rgba = cmap(norm)  # RGBA float in [0,1]
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)

    img_pil = Image.fromarray(rgb, mode="RGB")
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    img_arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    return img_arr.astype(np.float32)


def _audio_to_spectrogram_tf(file_path: str) -> np.ndarray:
    """TensorFlow-only preprocessing fallback.

    Note: This is kept for environments where librosa/matplotlib are unavailable.
    """
    if tf is None:
        raise ImportError("TensorFlow is required.")

    audio, _sr = _load_audio_mono(file_path)

    # Match librosa.stft(center=True, pad_mode='reflect') by padding n_fft//2.
    pad = N_FFT // 2
    n_dyn = tf.shape(audio)[0]
    audio = tf.cond(
        n_dyn > pad,
        lambda: tf.pad(audio, paddings=[[pad, pad]], mode="REFLECT"),
        lambda: tf.pad(audio, paddings=[[pad, pad]], mode="CONSTANT"),
    )

    stft = tf.signal.stft(
        audio,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    spec = tf.abs(stft) ** 2
    mel_w = _get_mel_weight_matrix()
    mel = tf.matmul(spec, mel_w)  # (frames, n_mels)
    mel = tf.transpose(mel)       # (n_mels, frames)

    # librosa.power_to_db(mel, ref=np.max, top_db=80)
    mel = tf.maximum(mel, 1e-10)
    log_mel = 10.0 * tf.math.log(mel) / tf.math.log(10.0)
    ref = tf.reduce_max(log_mel)
    mel_db = log_mel - ref
    mel_db = tf.maximum(mel_db, -TOP_DB)

    # Match origin='lower' by flipping vertically for image coordinates
    mel_db = tf.reverse(mel_db, axis=[0])

    mel_np = mel_db.numpy().astype(np.float32)
    vmin = float(np.min(mel_np))
    vmax = float(np.max(mel_np))
    denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1.0
    norm = (mel_np - vmin) / denom
    norm = np.clip(norm, 0.0, 1.0)

    cmap = _magma_like_colormap_256()
    idx = (norm * 255.0).astype(np.uint8)
    rgb = cmap[idx]  # (n_mels, frames, 3)

    img = (rgb * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(img, mode="RGB")
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    img_arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    return img_arr.astype(np.float32)


def _audio_to_spectrogram(file_path: str) -> np.ndarray:
    """Load a .wav file and convert it to a normalised 128×128 RGB-like array."""
    if librosa is not None and mpl_cm is not None:
        try:
            return _audio_to_spectrogram_librosa(file_path)
        except Exception:
            pass
    return _audio_to_spectrogram_tf(file_path)


def predict_audio(
    file_path: str,
    *,
    return_heatmap: bool = False,
) -> tuple[str, float] | tuple[str, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict whether an audio file is REAL or FAKE.

    Args:
        file_path: Path to the .wav audio file.

    Returns:
        label:      "REAL" or "FAKE"
        confidence: Probability score in [0.0, 1.0] for the predicted class.

        If return_heatmap=True, also returns:
            spectrogram_rgb: (128, 128, 3) float32 in [0,1]
            heatmap_rgb:     (128, 128, 3) uint8
            overlay_rgb:     (128, 128, 3) uint8

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

        confidence = round(confidence, 4)

        if not return_heatmap:
            return label, confidence

        class_index = 1 if label == "REAL" else 0
        heatmap = generate_gradcam(model=model, input_image=img_batch, class_index=class_index)
        heatmap_rgb, overlay_rgb = overlay_heatmap(image_rgb=img, heatmap=heatmap, alpha=0.45, colormap="jet")

        return label, confidence, img, heatmap_rgb, overlay_rgb

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
