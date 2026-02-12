# AGENTS.md - Wazo STT Server

## Project Overview

Python FastAPI server providing speech-to-text transcription using NVIDIA Parakeet model via onnx_asr. OpenAI Whisper API compatible with async job queue and SQLite persistence.

## Build/Lint/Test Commands

### Development Server
```bash
pip install -r requirements.txt    # Install dependencies
python main.py                      # Run server (default: http://0.0.0.0:8000)
```

### Docker
```bash
docker compose up -d --build       # Build and run container
docker build -t wazo-stt-server .  # Build image manually
```

### Testing
```bash
# Manual API testing
curl http://localhost:8000/health                                          # Health check
curl http://localhost:8000/v1/audio/transcriptions/status                 # Queue status
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@test.wav" -F "user_uuid=test" -F "message_id=test-001"        # Submit job
curl http://localhost:8000/v1/audio/transcriptions/{job_id}               # Get job status
```

No automated test framework is configured. Manual testing via curl or FastAPI docs at `/docs`.

## Code Style Guidelines

### Language and Comments

- **Code comments in English** for this Python backend
- Docstrings for all public functions using triple-quoted style
- Example:
  ```python
  def process_audio(audio_data: bytes, filename: str) -> tuple[AudioSegment, float]:
      """
      Process audio file: load, normalize, convert to 16kHz mono
      Returns (processed_audio, duration_seconds)
      """
  ```

### Imports Organization

Group imports in this order, separated by blank lines:
1. Standard library (alphabetical)
2. Third-party packages (alphabetical)
3. Local imports

```python
import asyncio
import logging
import os
from collections import deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydub import AudioSegment
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `process_audio`, `transcribe_audio` |
| Variables | snake_case | `audio_data`, `job_queue` |
| Constants | UPPER_SNAKE_CASE | `MODEL_NAME`, `DB_PATH`, `SAMPLE_RATE` |
| Classes | PascalCase | `Job`, `TranscriptionResult` |
| Enums | PascalCase, values UPPER | `JobStatus.QUEUED` |
| Endpoints | snake_case in URL | `/v1/audio/transcriptions/status` |

### Type Hints

Use Python type hints for function signatures:
```python
def db_find_by_id(job_id: str) -> Optional[dict]:
def process_audio(audio_data: bytes, filename: str) -> tuple[AudioSegment, float]:
async def fetch_audio_from_url(url: str) -> tuple[bytes, str]:
```

### Configuration

Use environment variables via `os.getenv()` with defaults:
```python
MODEL_NAME = os.getenv("MODEL_NAME", "nemo-parakeet-tdt-0.6b-v3")
PORT = int(os.getenv("PORT", "8000"))
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"
```

### Error Handling

- Use FastAPI `HTTPException` for API errors
- Log errors with `logger.error()` or `logger.warning()`
- Include descriptive error messages

```python
if not file and not url:
    raise HTTPException(status_code=400, detail="No audio file or URL provided")

try:
    audio_data, filename = await fetch_audio_from_url(url)
except Exception as e:
    raise HTTPException(status_code=400, detail=f"Failed to fetch audio from URL: {e}")
```

### Async Patterns

- Use `async`/`await` for I/O operations
- Use `asyncio.Condition` for coordination (job queue)
- Use `asynccontextmanager` for lifespan management

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize
    worker_task = asyncio.create_task(job_worker())
    yield
    # Cleanup
    worker_task.cancel()
```

### Database Operations

- Use sqlite3 directly (no ORM)
- Create connections per-operation (no connection pooling needed for SQLite)
- Use `sqlite3.Row` for dict-like access

```python
def db_find_by_id(job_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM transcriptions WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
```

### Global State

Use module-level globals with careful initialization:
```python
jobs: dict[str, Job] = {}
job_queue: deque[str] = deque()
job_queue_condition: asyncio.Condition = None
model = None  # onnx_asr model instance
```

### FastAPI Endpoints

Use typed parameters with `Query`, `Form`, `File`:
```python
@app.post("/v1/audio/transcriptions")
async def submit_transcription(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    user_uuid: str = Form(...),
    message_id: str = Form(...),
):
```

### File Structure

```
/
├── main.py              # Single-file application (FastAPI + workers)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── README.md            # Project documentation
└── API.md               # API documentation
```

### Logging

Use structured logging with module name:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wazo-stt-server")

logger.info(f"Job {job_id} completed: {len(result.text)} chars")
logger.error(f"Job {job_id} failed: {e}")
```

### API Response Format

Return consistent JSON responses:
```python
# Success with data
return {"job_id": job_id, "status": "queued", "queue_position": 1}

# Error via HTTPException
raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

# Accepted/processing
return JSONResponse(status_code=202, content={"status": "processing"})
```

### Cleanup Patterns

Use `try/finally` for resource cleanup:
```python
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(audio_data)
    temp_path = f.name

try:
    # Process audio
    result = transcribe_audio(audio)
finally:
    os.unlink(temp_path)
```

### Optional Dependencies

Handle optional imports gracefully:
```python
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    DiarizationPipeline = None
```