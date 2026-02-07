#!/usr/bin/env python3
"""
Wazo STT Server - Speech-to-Text API using NVIDIA Parakeet model
OpenAI Whisper API compatible endpoints with async job queue and SQLite persistence
"""

import asyncio
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import onnx_asr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydub import AudioSegment
from pydub.effects import normalize

# Optional pyannote for diarization
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    DiarizationPipeline = None

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "nemo-parakeet-tdt-0.6b-v3")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/parakeet")  # Local path to downloaded model
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DB_PATH = os.getenv("DB_PATH", "/data/transcriptions.db")
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))
MAX_AUDIO_DURATION_SECS = int(os.getenv("MAX_AUDIO_DURATION_SECS", "480"))  # 8 minutes
TEXT_RETENTION_DAYS = int(os.getenv("TEXT_RETENTION_DAYS", "365"))  # 1 year
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"
SAMPLE_RATE = 16000

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wazo-stt-server")


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscriptionResult:
    text: str
    segments: list = field(default_factory=list)
    duration: float = 0.0


@dataclass
class Job:
    id: str
    user_uuid: str
    message_id: str
    status: JobStatus
    audio_path: str
    queue_position: Optional[int] = None
    result: Optional[TranscriptionResult] = None
    error: Optional[str] = None
    created_at: int = 0


# Global state
jobs: dict[str, Job] = {}
job_queue: deque[str] = deque()
job_queue_condition: asyncio.Condition = None
model = None  # onnx_asr model instance
diarization_pipeline = None  # pyannote diarization pipeline
db_lock: asyncio.Lock = None


# Database functions
def init_database(db_path: str):
    """Initialize SQLite database"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id TEXT PRIMARY KEY,
            user_uuid TEXT NOT NULL,
            message_id TEXT NOT NULL,
            status TEXT NOT NULL,
            text TEXT,
            duration REAL,
            error TEXT,
            created_at INTEGER NOT NULL,
            completed_at INTEGER,
            last_lookup INTEGER,
            UNIQUE(user_uuid, message_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_message ON transcriptions(user_uuid, message_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_last_lookup ON transcriptions(last_lookup)")

    # Migration: add last_lookup column if it doesn't exist (for existing databases)
    try:
        conn.execute("ALTER TABLE transcriptions ADD COLUMN last_lookup INTEGER")
        logger.info("Added last_lookup column to existing database")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")

    # Run cleanup on startup
    cleanup_old_texts()


def cleanup_old_texts():
    """Remove text from transcriptions not accessed in TEXT_RETENTION_DAYS"""
    cutoff_time = int(time.time()) - (TEXT_RETENTION_DAYS * 24 * 60 * 60)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """UPDATE transcriptions
           SET text = NULL
           WHERE text IS NOT NULL
           AND (last_lookup IS NULL OR last_lookup < ?)
           AND created_at < ?""",
        (cutoff_time, cutoff_time)
    )
    cleaned = cursor.rowcount
    conn.commit()
    conn.close()
    if cleaned > 0:
        logger.info(f"Cleaned up text from {cleaned} old transcriptions (not accessed in {TEXT_RETENTION_DAYS} days)")


def db_insert(job: Job):
    """Insert a new job into the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO transcriptions (id, user_uuid, message_id, status, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (job.id, job.user_uuid, job.message_id, job.status.value, job.created_at)
    )
    conn.commit()
    conn.close()


def db_update_status(job_id: str, status: str):
    """Update job status in database"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE transcriptions SET status = ? WHERE id = ?", (status, job_id))
    conn.commit()
    conn.close()


def db_update_completed(job_id: str, text: str, duration: float):
    """Update job with completed transcription"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """UPDATE transcriptions SET status = ?, text = ?, duration = ?, completed_at = ?
           WHERE id = ?""",
        ("completed", text, duration, int(time.time()), job_id)
    )
    conn.commit()
    conn.close()


def db_update_failed(job_id: str, error: str):
    """Update job with failure"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE transcriptions SET status = ?, error = ? WHERE id = ?",
        ("failed", error, job_id)
    )
    conn.commit()
    conn.close()


def db_find_by_user_and_message(user_uuid: str, message_id: str, update_lookup: bool = True) -> Optional[dict]:
    """Find transcription by user_uuid and message_id"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM transcriptions WHERE user_uuid = ? AND message_id = ?",
        (user_uuid, message_id)
    )
    row = cursor.fetchone()

    # Update last_lookup timestamp if record found and update requested
    if row and update_lookup:
        conn.execute(
            "UPDATE transcriptions SET last_lookup = ? WHERE user_uuid = ? AND message_id = ?",
            (int(time.time()), user_uuid, message_id)
        )
        conn.commit()

    conn.close()
    return dict(row) if row else None


def db_find_by_id(job_id: str) -> Optional[dict]:
    """Find transcription by job ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM transcriptions WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def db_delete_by_user_and_message(user_uuid: str, message_id: str) -> bool:
    """Delete transcription by user_uuid and message_id"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "DELETE FROM transcriptions WHERE user_uuid = ? AND message_id = ?",
        (user_uuid, message_id)
    )
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def db_get_stats() -> dict:
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM transcriptions")
    total = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM transcriptions WHERE status = 'completed'")
    completed = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM transcriptions WHERE status = 'failed'")
    failed = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM transcriptions WHERE status IN ('queued', 'processing')")
    pending = cursor.fetchone()[0]
    conn.close()
    return {"total": total, "completed": completed, "failed": failed, "pending": pending}


# Audio processing
def process_audio(audio_data: bytes, filename: str) -> tuple[AudioSegment, float]:
    """
    Process audio file: load, normalize, convert to 16kHz mono
    Returns (processed_audio, duration_seconds)
    """
    # Save to temp file for pydub to read
    suffix = Path(filename).suffix if filename else ".wav"
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        temp_path = f.name

    try:
        # Load audio with pydub (handles format detection)
        audio = AudioSegment.from_file(temp_path)

        logger.info(f"Audio loaded: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels} channels")

        # Normalize audio (like HuggingFace space does)
        audio = normalize(audio)

        # Convert to mono 16kHz 16-bit (required for ASR)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit
        audio = audio.set_frame_rate(SAMPLE_RATE)

        duration_secs = len(audio) / 1000.0

        logger.info(f"Audio processed: {duration_secs:.2f}s at {SAMPLE_RATE}Hz mono")

        return audio, duration_secs
    finally:
        os.unlink(temp_path)


def transcribe_audio(audio: AudioSegment) -> TranscriptionResult:
    """Transcribe audio using onnx_asr"""
    global model

    # Save processed audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio.export(f.name, format="wav")
        temp_path = f.name

    try:
        # Transcribe with timestamps using recognize method
        # With VAD enabled, recognize() returns a generator that yields segments
        result = model.recognize(temp_path)

        # Collect all segments from the generator
        segments = []
        all_text = []

        # Iterate over the generator (VAD returns multiple segments)
        for seg in result:
            if hasattr(seg, 'text'):
                all_text.append(seg.text)
                segments.append({
                    "id": len(segments),
                    "start": seg.start if hasattr(seg, 'start') else 0,
                    "end": seg.end if hasattr(seg, 'end') else 0,
                    "text": seg.text
                })
            elif hasattr(seg, 'segments') and seg.segments:
                # Handle nested segments
                for s in seg.segments:
                    all_text.append(s.text if hasattr(s, 'text') else str(s))
                    segments.append({
                        "id": len(segments),
                        "start": s.start if hasattr(s, 'start') else 0,
                        "end": s.end if hasattr(s, 'end') else 0,
                        "text": s.text if hasattr(s, 'text') else str(s)
                    })

        text = " ".join(all_text)
        duration = segments[-1]["end"] if segments else len(audio) / 1000.0

        return TranscriptionResult(text=text, segments=segments, duration=duration)
    finally:
        os.unlink(temp_path)


def diarize_audio(audio_path: str) -> list[dict]:
    """Perform speaker diarization on audio file"""
    global diarization_pipeline

    if not diarization_pipeline:
        return []

    try:
        diarization = diarization_pipeline(audio_path)
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        return speakers
    except Exception as e:
        logger.warning(f"Diarization failed: {e}")
        return []


def merge_transcription_with_diarization(
    segments: list[dict],
    diarization: list[dict]
) -> list[dict]:
    """Merge transcription segments with speaker diarization"""
    if not diarization:
        return segments

    merged = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_mid = (seg_start + seg_end) / 2

        # Find the speaker active at the segment midpoint
        speaker = "UNKNOWN"
        for d in diarization:
            if d["start"] <= seg_mid <= d["end"]:
                speaker = d["speaker"]
                break

        merged.append({
            **seg,
            "speaker": speaker
        })

    return merged


async def fetch_audio_from_url(url: str) -> tuple[bytes, str]:
    """Fetch audio from URL"""
    async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()

        # Try to get filename from Content-Disposition or URL
        filename = "audio.wav"
        if "content-disposition" in response.headers:
            cd = response.headers["content-disposition"]
            if "filename=" in cd:
                filename = cd.split("filename=")[1].strip('"')
        else:
            url_path = url.split("?")[0]
            if "/" in url_path:
                filename = url_path.split("/")[-1] or "audio.wav"

        logger.info(f"Fetched {len(response.content)} bytes from URL, filename: {filename}")
        return response.content, filename


# Background worker
async def job_worker():
    """Background worker that processes transcription jobs"""
    global model, diarization_pipeline, job_queue_condition

    logger.info("Job worker starting, loading model...")

    try:
        # Load model with CPU provider, VAD, and timestamps
        # Use local path as second argument to load from/cache to that directory
        logger.info(f"Loading model {MODEL_NAME} from: {MODEL_PATH}")
        base_model = onnx_asr.load_model(MODEL_NAME, MODEL_PATH, providers=["CPUExecutionProvider"])
        # Load Silero VAD separately (required in onnx_asr 0.10+)
        vad = onnx_asr.load_vad("silero")
        model = base_model.with_vad(vad).with_timestamps()
        logger.info(f"Model {MODEL_NAME} loaded successfully with Silero VAD")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load diarization pipeline if enabled
    if ENABLE_DIARIZATION:
        if PYANNOTE_AVAILABLE:
            try:
                logger.info("Loading pyannote diarization pipeline...")
                diarization_pipeline = DiarizationPipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
                logger.info("Diarization pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load diarization pipeline: {e}")
                logger.warning("Diarization will be disabled")
        else:
            logger.warning("pyannote not installed, diarization disabled")

    logger.info("Job worker ready, waiting for jobs...")

    while True:
        job_id = None

        # Wait for a job
        async with job_queue_condition:
            while not job_queue:
                await job_queue_condition.wait()
            job_id = job_queue.popleft()

        if job_id not in jobs:
            continue

        job = jobs[job_id]
        logger.info(f"Processing job: {job_id}")

        try:
            # Update status
            job.status = JobStatus.PROCESSING
            job.queue_position = None
            db_update_status(job_id, "processing")

            # Update queue positions for remaining jobs
            for i, qid in enumerate(job_queue):
                if qid in jobs:
                    jobs[qid].queue_position = i + 1

            # Load and process audio
            with open(job.audio_path, "rb") as f:
                audio_data = f.read()

            audio, duration = process_audio(audio_data, job.audio_path)

            # Check duration limit
            if duration > MAX_AUDIO_DURATION_SECS:
                raise ValueError(
                    f"Audio too long: {duration/60:.1f} minutes. "
                    f"Maximum allowed: {MAX_AUDIO_DURATION_SECS/60:.0f} minutes."
                )

            # Transcribe
            result = transcribe_audio(audio)

            job.result = result
            job.status = JobStatus.COMPLETED
            db_update_completed(job_id, result.text, result.duration)

            logger.info(f"Job {job_id} completed: {len(result.text)} chars")

        except Exception as e:
            error_msg = str(e)
            job.error = error_msg
            job.status = JobStatus.FAILED
            db_update_failed(job_id, error_msg)
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            # Clean up temp audio file
            try:
                if job.audio_path and os.path.exists(job.audio_path):
                    os.unlink(job.audio_path)
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global job_queue_condition, db_lock

    # Initialize
    job_queue_condition = asyncio.Condition()
    db_lock = asyncio.Lock()
    init_database(DB_PATH)

    # Start worker
    worker_task = asyncio.create_task(job_worker())

    yield

    # Cleanup
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


# FastAPI app
app = FastAPI(
    title="Wazo STT Server",
    description="OpenAI Whisper API compatible Speech-to-Text server using NVIDIA Parakeet",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = db_get_stats()
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": "1.0.0",
        "queue_length": len(job_queue),
        "db_stats": stats
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "parakeet-tdt",
                "object": "model",
                "created": 1699000000,
                "owned_by": "nvidia"
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1699000000,
                "owned_by": "openai"
            }
        ]
    }


@app.get("/v1/audio/transcriptions/status")
async def queue_status():
    """Get queue status"""
    processing = any(j.status == JobStatus.PROCESSING for j in jobs.values())
    return {
        "queue_length": len(job_queue),
        "processing": processing,
        "max_queue_size": MAX_QUEUE_SIZE
    }


@app.get("/v1/audio/transcriptions/lookup")
async def lookup_transcription(user_uuid: str = Query(...), message_id: str = Query(...)):
    """Lookup existing transcription by user_uuid and message_id"""
    record = db_find_by_user_and_message(user_uuid, message_id)

    if record:
        return {
            "found": True,
            "job_id": record["id"],
            "status": record["status"],
            "text": record["text"],
            "duration": record["duration"],
            "error": record["error"],
            "created_at": record["created_at"]
        }

    return {"found": False}


@app.post("/v1/audio/transcriptions")
async def submit_transcription(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    user_uuid: str = Form(...),
    message_id: str = Form(...),
    force: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
):
    """Submit audio for transcription"""
    global job_queue_condition

    force_retranscribe = force in ("true", "1", True)

    # Check for existing transcription
    existing = db_find_by_user_and_message(user_uuid, message_id)
    if existing:
        if force_retranscribe:
            logger.info(f"Force re-transcription requested for user={user_uuid}, message={message_id}")
            db_delete_by_user_and_message(user_uuid, message_id)
            # Also remove from in-memory if present
            for job_id, job in list(jobs.items()):
                if job.user_uuid == user_uuid and job.message_id == message_id:
                    del jobs[job_id]
        else:
            logger.info(f"Found existing transcription for user={user_uuid}, message={message_id}")
            return {
                "job_id": existing["id"],
                "status": existing["status"],
                "message": "Transcription already exists",
                "cached": True
            }

    # Get audio data
    if file:
        audio_data = await file.read()
        filename = file.filename or "audio.wav"
        logger.info(f"Received file upload: {filename}, {len(audio_data)} bytes")
    elif url:
        logger.info(f"Fetching audio from URL: {url}")
        try:
            audio_data, filename = await fetch_audio_from_url(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch audio from URL: {e}")
    else:
        raise HTTPException(status_code=400, detail="No audio file or URL provided")

    # Check queue size
    if len(job_queue) >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=429,
            detail=f"Queue is full ({MAX_QUEUE_SIZE} jobs). Please try again later."
        )

    logger.info(f"Received transcription request: user={user_uuid}, message={message_id}, filename={filename}, size={len(audio_data)} bytes")

    # Save audio to temp file
    suffix = Path(filename).suffix if filename else ".wav"
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        audio_path = f.name

    # Create job
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        user_uuid=user_uuid,
        message_id=message_id,
        status=JobStatus.QUEUED,
        audio_path=audio_path,
        created_at=int(time.time())
    )

    # Insert into database
    db_insert(job)

    # Add to queue
    jobs[job_id] = job
    job_queue.append(job_id)
    job.queue_position = len(job_queue)

    # Notify worker
    async with job_queue_condition:
        job_queue_condition.notify()

    logger.info(f"Job {job_id} queued at position {job.queue_position}")

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": job.queue_position,
        "message": f"Job queued. Position in queue: {job.queue_position}",
        "cached": False
    }


@app.get("/v1/audio/transcriptions/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    # Check in-memory first
    if job_id in jobs:
        job = jobs[job_id]
        return {
            "job_id": job.id,
            "status": job.status.value,
            "queue_position": job.queue_position,
            "user_uuid": job.user_uuid,
            "message_id": job.message_id,
            "text": job.result.text if job.result else None,
            "duration": job.result.duration if job.result else None,
            "error": job.error,
            "created_at": job.created_at
        }

    # Fall back to database
    record = db_find_by_id(job_id)
    if record:
        return {
            "job_id": record["id"],
            "status": record["status"],
            "queue_position": None,
            "user_uuid": record["user_uuid"],
            "message_id": record["message_id"],
            "text": record["text"],
            "duration": record["duration"],
            "error": record["error"],
            "created_at": record["created_at"]
        }

    raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@app.get("/v1/audio/transcriptions/{job_id}/result")
async def get_job_result(job_id: str, response_format: str = Query(default="json")):
    """Get transcription result"""
    # Check in-memory first
    job = jobs.get(job_id)
    record = None

    if not job:
        record = db_find_by_id(job_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Get status and result
    if job:
        status = job.status
        text = job.result.text if job.result else None
        error = job.error
        duration = job.result.duration if job.result else None
        segments = job.result.segments if job.result else []
    else:
        status = JobStatus(record["status"])
        text = record["text"]
        error = record["error"]
        duration = record["duration"]
        segments = []

    # Handle different statuses
    if status == JobStatus.COMPLETED:
        if response_format == "text":
            return PlainTextResponse(text or "")
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": "en",
                "duration": duration or 0,
                "text": text or "",
                "segments": segments
            }
        elif response_format == "srt":
            srt = ""
            for i, seg in enumerate(segments):
                start = format_srt_time(seg.get("start", 0))
                end = format_srt_time(seg.get("end", 0))
                srt += f"{i+1}\n{start} --> {end}\n{seg.get('text', '')}\n\n"
            return PlainTextResponse(srt, media_type="text/plain")
        elif response_format == "vtt":
            vtt = "WEBVTT\n\n"
            for seg in segments:
                start = format_vtt_time(seg.get("start", 0))
                end = format_vtt_time(seg.get("end", 0))
                vtt += f"{start} --> {end}\n{seg.get('text', '')}\n\n"
            return PlainTextResponse(vtt, media_type="text/vtt")
        else:
            return {"text": text or ""}

    elif status == JobStatus.FAILED:
        raise HTTPException(status_code=400, detail=error or "Transcription failed")

    elif status in (JobStatus.QUEUED, JobStatus.PROCESSING):
        queue_pos = job.queue_position if job else None
        msg = f"Job is still queued at position {queue_pos}" if queue_pos else "Job is still processing"
        return JSONResponse(status_code=202, content={"status": status.value, "message": msg})

    raise HTTPException(status_code=500, detail="Unknown job status")


@app.post("/v1/audio/recordings")
async def transcribe_recording(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    diarize: Optional[str] = Form("true"),
    response_format: Optional[str] = Form("json"),
):
    """
    Transcribe a call recording with optional speaker diarization.
    This is a synchronous endpoint - returns result directly.
    """
    global model, diarization_pipeline

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    enable_diarization = diarize in ("true", "1", True)

    # Get audio data
    if file:
        audio_data = await file.read()
        filename = file.filename or "audio.wav"
        logger.info(f"Recording: received file {filename}, {len(audio_data)} bytes")
    elif url:
        logger.info(f"Recording: fetching from URL")
        try:
            audio_data, filename = await fetch_audio_from_url(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch audio: {e}")
    else:
        raise HTTPException(status_code=400, detail="No audio file or URL provided")

    try:
        # Process audio
        audio, duration = process_audio(audio_data, filename)

        if duration > MAX_AUDIO_DURATION_SECS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {duration/60:.1f} min. Max: {MAX_AUDIO_DURATION_SECS/60:.0f} min."
            )

        # Save processed audio for transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio.export(f.name, format="wav")
            temp_path = f.name

        try:
            # Transcribe - with VAD enabled, recognize() returns a generator
            result = model.recognize(temp_path)

            segments = []
            all_text = []

            # Iterate over the generator (VAD returns multiple segments)
            for seg in result:
                if hasattr(seg, 'text'):
                    all_text.append(seg.text)
                    segments.append({
                        "id": len(segments),
                        "start": seg.start if hasattr(seg, 'start') else 0,
                        "end": seg.end if hasattr(seg, 'end') else 0,
                        "text": seg.text
                    })
                elif hasattr(seg, 'segments') and seg.segments:
                    for s in seg.segments:
                        all_text.append(s.text if hasattr(s, 'text') else str(s))
                        segments.append({
                            "id": len(segments),
                            "start": s.start if hasattr(s, 'start') else 0,
                            "end": s.end if hasattr(s, 'end') else 0,
                            "text": s.text if hasattr(s, 'text') else str(s)
                        })

            text = " ".join(all_text)

            # Perform diarization if enabled and available
            speakers = []
            if enable_diarization and diarization_pipeline:
                logger.info("Performing speaker diarization...")
                speakers = diarize_audio(temp_path)
                segments = merge_transcription_with_diarization(segments, speakers)
                logger.info(f"Found {len(set(s['speaker'] for s in speakers))} speakers")

        finally:
            os.unlink(temp_path)

        # Format response
        if response_format == "text":
            if segments and any("speaker" in s for s in segments):
                # Format with speaker labels
                lines = []
                current_speaker = None
                for seg in segments:
                    speaker = seg.get("speaker", "UNKNOWN")
                    if speaker != current_speaker:
                        lines.append(f"\n[{speaker}]")
                        current_speaker = speaker
                    lines.append(seg.get("text", ""))
                return PlainTextResponse("\n".join(lines))
            return PlainTextResponse(text)

        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "duration": duration,
                "text": text,
                "segments": segments,
                "speakers": list(set(s.get("speaker") for s in segments if "speaker" in s)),
                "diarization_enabled": enable_diarization and diarization_pipeline is not None
            }

        else:  # json
            return {
                "text": text,
                "segments": segments if segments else None,
                "duration": duration,
                "diarization_enabled": enable_diarization and diarization_pipeline is not None
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recording transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/recordings/status")
async def recording_status():
    """Get recording endpoint status"""
    return {
        "available": model is not None,
        "diarization_available": diarization_pipeline is not None,
        "diarization_model": "pyannote/speaker-diarization-3.1" if diarization_pipeline else None
    }


def format_srt_time(seconds: float) -> str:
    """Format time for SRT subtitles"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    """Format time for VTT subtitles"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
