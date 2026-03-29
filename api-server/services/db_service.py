from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PredictionLog:
    filename: str
    prediction: str
    confidence: float
    timestamp: str


class DBService:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def init(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename   TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp  TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")

    def log_prediction(self, *, filename: str, prediction: str, confidence: float) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions (filename, prediction, confidence, timestamp) VALUES (?, ?, ?, ?)",
                (filename, prediction, float(confidence), ts),
            )

    def get_history(self) -> list[PredictionLog]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT filename, prediction, confidence, timestamp FROM predictions ORDER BY timestamp DESC"
            )
            rows = cur.fetchall()

        return [
            PredictionLog(
                filename=str(r[0]),
                prediction=str(r[1]),
                confidence=float(r[2]),
                timestamp=str(r[3]),
            )
            for r in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


def default_db_path() -> Path:
    """Get DB path from env or default to api-server/predictions.sqlite3."""

    env = os.getenv("PREDICTIONS_DB_PATH", "").strip()
    if env:
        return Path(env)
    # Default to ./predictions.sqlite3 next to this file's parent folder
    return Path(__file__).resolve().parents[1] / "predictions.sqlite3"
