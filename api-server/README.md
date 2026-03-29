# API Server (FastAPI)

## Run

From this folder:

```bash
uvicorn main:app --reload
```

Or from the project root:

```bash
uvicorn main:app --reload --app-dir api-server
```

## Endpoint

### `POST /predict`

- **Input**: multipart form upload field `file` (any common audio format)
- **Output**:

```json
{
  "prediction": "REAL",
  "confidence": 0.92
}
```

### `GET /history`

Returns an array of logged predictions (most recent first).

```json
[
  {
    "filename": "example.wav",
    "prediction": "REAL",
    "confidence": 0.92,
    "timestamp": "2026-03-29T09:38:50Z"
  }
]
```

## Storage (SQLite)

- Default DB file: `api-server/predictions.sqlite3`
- Override with env var: `PREDICTIONS_DB_PATH`
