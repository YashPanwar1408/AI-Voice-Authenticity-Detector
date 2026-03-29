from __future__ import annotations

import io
import os
import sys
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

import librosa
import librosa.display
from imageio_ffmpeg import get_ffmpeg_exe


# Ensure project root is importable so `import src.*` works even when Streamlit
# is launched from inside the `app/` directory.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HISTORY_ENDPOINT = f"{API_BASE_URL}/history"


st.set_page_config(page_title="AI Voice Detector", layout="wide")


def call_api_predict(*, uploaded_bytes: bytes, filename: str, content_type: str | None) -> dict[str, Any]:
    """Call backend POST /predict and return parsed JSON."""
    try:
        resp = requests.post(
            PREDICT_ENDPOINT,
            files={
                "file": (filename, uploaded_bytes, content_type or "application/octet-stream"),
            },
            timeout=120,
        )
    except requests.RequestException:
        raise RuntimeError(
            "Backend API is not reachable. Start it with: uvicorn main:app --reload --app-dir api-server"
        )

    if resp.status_code != 200:
        try:
            payload = resp.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
        except Exception:
            detail = None
        message = detail or (resp.text.strip() if resp.text else None) or f"HTTP {resp.status_code}"
        raise RuntimeError(message)

    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("API returned invalid JSON")
    if data.get("prediction") not in ("REAL", "FAKE"):
        raise RuntimeError("API returned an invalid prediction")
    try:
        data["confidence"] = float(data.get("confidence"))
    except Exception:
        raise RuntimeError("API returned an invalid confidence")
    return data


def load_history() -> pd.DataFrame:
    """Call backend GET /history and return a DataFrame."""
    try:
        resp = requests.get(HISTORY_ENDPOINT, timeout=30)
    except requests.RequestException:
        raise RuntimeError(
            "Backend API is not reachable. Start it with: uvicorn main:app --reload --app-dir api-server"
        )

    if resp.status_code != 200:
        raise RuntimeError(f"History request failed (HTTP {resp.status_code})")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("API returned invalid history JSON")

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["filename", "prediction", "confidence", "timestamp"])

    for col in ("filename", "prediction", "timestamp"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False)

    return df


def _convert_to_wav(uploaded_bytes: bytes) -> bytes:
    """Convert audio to PCM S16LE WAV bytes (16kHz mono) using bundled ffmpeg."""
    ffmpeg = get_ffmpeg_exe()

    tmp_path: str | None = None
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
            subprocess.run(
                cmd,
                input=uploaded_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg conversion failed: {err}")

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


def _wav_stats(wav_bytes: bytes) -> tuple[float | None, float | None, int | None, float | None, int | None]:
    """Return (rms, peak, sample_rate, duration_seconds, channels) when possible."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = int(wf.getframerate())
            channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            nframes = int(wf.getnframes())
            raw = wf.readframes(nframes)

        if sampwidth != 2 or sr <= 0 or nframes <= 0:
            return None, None, sr or None, (nframes / sr) if sr else None, channels or None

        x = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            x = x.reshape(-1, channels)
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


def plot_waveform(wav_bytes: bytes) -> plt.Figure:
    """Waveform plot using librosa + matplotlib."""
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=16000, mono=True)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_spectrogram(wav_bytes: bytes) -> plt.Figure:
    """Mel spectrogram visualization using librosa + matplotlib."""
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Mel Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def _render_single_prediction_page() -> None:
    st.header("Single Prediction")
    st.caption("Upload one audio file and run a single prediction via the backend API.")
    st.divider()

    col_left, col_right = st.columns([1.2, 1.0], gap="large")

    with col_left:
        st.subheader("Upload")
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "webm", "mp4"],
        )

        uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        wav_bytes: bytes | None = None

        if uploaded_bytes:
            try:
                wav_bytes = _convert_to_wav(uploaded_bytes)
            except Exception as e:
                st.error(f"Could not decode/convert '{uploaded_file.name}': {e}")
                return

            st.audio(wav_bytes, format="audio/wav")
            rms, peak, sr, duration, channels = _wav_stats(wav_bytes)
            if duration is not None and sr is not None:
                ch_text = f" ({channels}ch)" if channels else ""
                st.caption(f"Audio: ~{duration:.1f}s @ {sr} Hz{ch_text}")
            if duration is not None and duration < 5.0:
                st.caption("Note: clips shorter than ~5s may be looped in preprocessing.")
            if peak is not None and peak < 1e-3:
                st.warning(f"This audio looks like silence (peak={peak:.6f}).")

    with col_right:
        st.subheader("Prediction")
        if uploaded_file is None:
            st.info("Upload a file to enable prediction.")
        else:
            if st.button("Run Prediction", type="primary", use_container_width=True):
                with st.spinner("Calling backend and running inference..."):
                    try:
                        result = call_api_predict(
                            uploaded_bytes=uploaded_bytes,
                            filename=uploaded_file.name,
                            content_type=getattr(uploaded_file, "type", None),
                        )
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        return

                prediction = str(result["prediction"])
                confidence = float(result["confidence"])

                st.divider()
                if prediction == "REAL":
                    st.success("Prediction: REAL")
                else:
                    st.error("Prediction: FAKE")

                st.write(f"Confidence: {confidence:.2%}")
                st.progress(min(max(confidence, 0.0), 1.0))

                if wav_bytes is not None:
                    try:
                        from src.predict_cnn import predict_audio

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(wav_bytes)
                            tmp_path = tmp.name

                        try:
                            _, _, spect_rgb, heatmap_rgb, overlay_rgb = predict_audio(
                                tmp_path,
                                return_heatmap=True,
                            )
                        finally:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

                        st.divider()
                        st.subheader("Grad-CAM Explainability")
                        i1, i2, i3 = st.columns(3, gap="large")
                        with i1:
                            st.caption("Original spectrogram (model input)")
                            st.image(spect_rgb, clamp=True, use_container_width=True)
                        with i2:
                            st.caption("Grad-CAM heatmap")
                            st.image(heatmap_rgb, use_container_width=True)
                        with i3:
                            st.caption("Overlay")
                            st.image(overlay_rgb, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM could not be generated: {e}")

    if uploaded_file is not None and uploaded_bytes is not None and wav_bytes is not None:
        st.divider()
        st.subheader("Audio Visualizations")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            try:
                fig = plot_waveform(wav_bytes)
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"Could not render waveform: {e}")
        with c2:
            try:
                fig = plot_spectrogram(wav_bytes)
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"Could not render spectrogram: {e}")


def _render_batch_prediction_page() -> None:
    st.header("Batch Prediction")
    st.caption("Upload multiple files, run predictions, and export results.")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload multiple audio files",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "webm", "mp4"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more files to start batch prediction.")
        return

    if st.button("Run Batch", type="primary"):
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        with st.spinner("Running batch predictions..."):
            for f in uploaded_files:
                try:
                    b = f.getvalue()
                    r = call_api_predict(
                        uploaded_bytes=b,
                        filename=f.name,
                        content_type=getattr(f, "type", None),
                    )
                    results.append(
                        {
                            "filename": f.name,
                            "prediction": r["prediction"],
                            "confidence": float(r["confidence"]),
                        }
                    )
                except Exception as e:
                    errors.append(f"{f.name}: {e}")

        if errors:
            st.warning("Some files failed to process:")
            for msg in errors:
                st.write(f"- {msg}")

        if not results:
            st.error("No predictions were generated.")
            return

        df = pd.DataFrame(results)
        df["confidence"] = df["confidence"].map(lambda x: round(float(x), 4))

        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


def _render_analytics_dashboard_page() -> None:
    st.header("Analytics Dashboard")
    st.caption("Aggregate statistics and charts from the backend prediction history.")
    st.divider()

    try:
        df = load_history()
    except Exception as e:
        st.error(f"Could not load history: {e}")
        st.info("Tip: start the backend API and try again.")
        return

    total = int(len(df))
    real_count = int((df.get("prediction") == "REAL").sum()) if total else 0
    fake_count = int((df.get("prediction") == "FAKE").sum()) if total else 0

    c1, c2, c3 = st.columns(3, gap="large")
    c1.metric("Total Predictions", f"{total}")
    c2.metric("REAL", f"{real_count}")
    c3.metric("FAKE", f"{fake_count}")

    st.divider()

    if total == 0:
        st.info("No history yet. Run a few predictions first.")
        return

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("REAL vs FAKE")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.pie(
            [real_count, fake_count],
            labels=["REAL", "FAKE"],
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
        )
        ax.axis("equal")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Confidence Distribution")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.histplot(df["confidence"].dropna(), bins=20, kde=True, ax=ax)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("Recent Predictions")
    recent = df.head(10).copy()
    if "timestamp" in recent.columns:
        recent["timestamp"] = recent["timestamp"].astype(str)
    if "confidence" in recent.columns:
        recent["confidence"] = recent["confidence"].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
    st.dataframe(recent[[c for c in ["timestamp", "filename", "prediction", "confidence"] if c in recent.columns]], use_container_width=True)


st.title("AI Voice Detector")
st.caption("Frontend analytics dashboard powered by a FastAPI inference backend.")

tab_single, tab_batch, tab_analytics = st.tabs(["Single Prediction", "Batch Prediction", "Analytics Dashboard"])

with tab_single:
    _render_single_prediction_page()

with tab_batch:
    _render_batch_prediction_page()

with tab_analytics:
    _render_analytics_dashboard_page()
