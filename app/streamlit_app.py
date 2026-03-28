import os
import sys
import tempfile
import io
import subprocess

import streamlit as st
import numpy as np
import soundfile as sf
from imageio_ffmpeg import get_ffmpeg_exe

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict_cnn import predict_audio


st.set_page_config(page_title="AI Voice Authenticity Detector", page_icon="🎙️", layout="wide")

st.title("AI Voice Authenticity Detector")
st.caption("Upload an audio file. The app converts it to WAV, then analyzes it.")


def _analyze_audio(audio_wav_bytes: bytes) -> tuple[str, float]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_wav_bytes)
        temp_path = tmp.name
    try:
        return predict_audio(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _wav_stats(
    wav_bytes: bytes,
) -> tuple[float | None, float | None, int | None, float | None, int | None]:
    """Return (rms, peak, sample_rate, duration_seconds, channels) when possible."""
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)

        # data shape: (n_samples, n_channels)
        channels = int(data.shape[1])
        channel_rms = np.sqrt(np.mean(np.square(data), axis=0))
        best_channel = int(np.argmax(channel_rms))
        mono = data[:, best_channel]

        rms = float(channel_rms[best_channel])
        peak = float(np.max(np.abs(mono)))
        duration = float(len(mono) / sr) if sr else None
        return rms, peak, int(sr), duration, channels
    except Exception:
        return None, None, None, None, None


def _convert_to_wav(uploaded_bytes: bytes, input_ext: str) -> bytes:
    """Convert common audio formats to WAV bytes.

    Uses a bundled `ffmpeg` executable via `imageio-ffmpeg`.
    """
    ext = (input_ext or "").lower().lstrip(".")
    if ext == "wav":
        return uploaded_bytes

    ffmpeg = get_ffmpeg_exe()

    # Decode from stdin and write WAV to stdout.
    # Preserve channel count; downstream will select the best channel.
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=uploaded_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg conversion failed: {err.strip()}")

    if not proc.stdout:
        raise RuntimeError("ffmpeg conversion produced empty output")

    return proc.stdout



with st.container(border=True):
    st.subheader("Upload")
    st.caption("Supported: wav, mp3, m4a, flac, ogg, aac, wma, webm, mp4")

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "webm", "mp4"],
        label_visibility="collapsed",
    )

    uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    uploaded_ext = (uploaded_file.name.split(".")[-1] if uploaded_file and "." in uploaded_file.name else "").lower()

    wav_bytes: bytes | None = None
    if uploaded_bytes:
        try:
            wav_bytes = _convert_to_wav(uploaded_bytes, uploaded_ext)
        except Exception as e:
            st.error(f"Could not decode/convert '{uploaded_file.name}': {e}")
            st.stop()

        st.audio(wav_bytes, format="audio/wav")
        rms, peak, sr, duration, channels = _wav_stats(wav_bytes)
        if duration is not None and sr is not None:
            ch_text = f" ({channels}ch)" if channels else ""
            st.caption(f"Audio: ~{duration:.1f}s @ {sr} Hz{ch_text}")
        if duration is not None and duration < 5.0:
            st.caption("Note: clips shorter than 5s are looped during analysis for better accuracy.")
        if peak is not None and peak < 1e-3:
            st.warning(f"This audio looks like silence (peak={peak:.6f}).")


st.divider()

with st.container(border=True):
    st.subheader("Analyze")

    if wav_bytes is None:
        st.info("Upload an audio file to enable analysis.")

    if st.button("Analyze", type="primary", use_container_width=True, disabled=wav_bytes is None):
        assert wav_bytes is not None
        with st.spinner("Analyzing..."):
            try:
                label, confidence = _analyze_audio(wav_bytes)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.subheader("📊 Result")
        if label == "REAL":
            st.success(f"✅ REAL voice ({confidence * 100:.2f}%)")
        else:
            st.error(f"🚨 FAKE voice ({confidence * 100:.2f}%)")
