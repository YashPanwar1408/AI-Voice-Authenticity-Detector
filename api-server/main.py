from __future__ import annotations

import os
import sys
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from imageio_ffmpeg import get_ffmpeg_exe


# Ensure project root is on sys.path so `import src.*` works
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from src import predict_cnn  # noqa: E402
from services.db_service import DBService, PredictionLog, default_db_path  # noqa: E402


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


class HistoryItem(BaseModel):
    filename: str
    prediction: str
    confidence: float
    timestamp: str


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="AI Voice Authenticity Detector API")

_db = DBService(default_db_path())

_cors_origins = _cors_origins_from_env()
_allow_credentials = False if _cors_origins == ["*"] else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    try:
        _db.init()
    except Exception:
        # Don't crash the server on startup; surface error when used.
        pass

    # Warm model cache early so first request isn't slow.
    # (Uses existing loader inside src.predict_cnn.)
    try:
        predict_cnn._get_model()  # type: ignore[attr-defined]
    except Exception:
        # Don't crash the server on startup; surface error on first request.
        pass


def _convert_to_pcm_wav(uploaded_bytes: bytes) -> bytes:
    """Convert arbitrary audio bytes to PCM S16LE WAV bytes at 16kHz mono.

    Uses a bundled `ffmpeg` executable via `imageio-ffmpeg`.

    Notes:
    - TensorFlow's `decode_wav` rejects WAV files with an unknown-length header.
      Writing to a real temp file ensures a correct RIFF header.
    """

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
            raise RuntimeError(f"ffmpeg conversion failed: {err or 'unknown error'}")

        wav_bytes = Path(tmp_path).read_bytes()
        if not wav_bytes:
            raise RuntimeError("ffmpeg conversion produced empty output")
        return wav_bytes

    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _predict_from_wav_bytes(wav_bytes: bytes) -> tuple[str, float]:
    tmp_audio_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp_audio_path = tmp.name

        label, confidence = predict_cnn.predict_audio(tmp_audio_path)
        return label, float(confidence)

    finally:
        if tmp_audio_path is not None:
            try:
                Path(tmp_audio_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        uploaded_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e}")

    if not uploaded_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        wav_bytes = _convert_to_pcm_wav(uploaded_bytes)
        label, confidence = _predict_from_wav_bytes(wav_bytes)

        try:
            _db.log_prediction(
                filename=file.filename,
                prediction=label,
                confidence=confidence,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save prediction: {e}")

        return PredictResponse(prediction=label, confidence=round(confidence, 4))

    except RuntimeError as e:
        # Conversion / decoding issues
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/history", response_model=list[HistoryItem])
def history() -> list[HistoryItem]:
    try:
        items: list[PredictionLog] = _db.get_history()
        return [
            HistoryItem(
                filename=i.filename,
                prediction=i.prediction,
                confidence=round(float(i.confidence), 4),
                timestamp=i.timestamp,
            )
            for i in items
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load history: {e}")