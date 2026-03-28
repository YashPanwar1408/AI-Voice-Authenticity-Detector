import os
import sys
import tempfile
import io
import subprocess
import wave

import streamlit as st
import numpy as np
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
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = int(wf.getframerate())
            channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            nframes = int(wf.getnframes())
            raw = wf.readframes(nframes)

        if sampwidth != 2 or sr <= 0 or nframes <= 0:
            # We intentionally export PCM S16LE in ffmpeg conversion.
            return None, None, sr or None, (nframes / sr) if sr else None, channels or None

        x = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            x = x.reshape(-1, channels)
            # pick loudest channel
            channel_rms = np.sqrt(np.mean(np.square(x.astype(np.float32)), axis=0))
            best_channel = int(np.argmax(channel_rms))
            x = x[:, best_channel]

        xf = x.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(xf))))
        peak = float(np.max(np.abs(xf)))
        duration = float(len(xf) / sr) if sr else None
        return rms, peak, sr, duration, channels
    except Exception:
        return None, None, None, None, None


def _convert_to_wav(uploaded_bytes: bytes) -> bytes:
    """Convert audio to PCM S16LE WAV bytes.

    Uses a bundled `ffmpeg` executable via `imageio-ffmpeg`.
    Always converts (even if the input is WAV) to ensure TensorFlow can decode it.
    """

    ffmpeg = get_ffmpeg_exe()

    # Important:
    # Writing WAV to stdout (pipe) makes ffmpeg emit an "unknown length" WAV
    # header with a 0xFFFFFFFF data chunk size. TensorFlow's decode_wav rejects
    # that. So we write a temp .wav file to get a proper RIFF header.

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            tmp_path,
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

        wav_bytes = open(tmp_path, "rb").read()
        if not wav_bytes:
            raise RuntimeError("ffmpeg conversion produced empty output")
        return wav_bytes

    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass



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
            wav_bytes = _convert_to_wav(uploaded_bytes)
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
