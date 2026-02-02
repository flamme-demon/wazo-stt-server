mod db;

use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use db::{Database, TranscriptionRecord};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    env,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, Mutex, RwLock};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{error, info, warn};
use uuid::Uuid;

const DEFAULT_MODEL_PATH: &str = "/models/parakeet";
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8000";
const DEFAULT_DB_PATH: &str = "/data/transcriptions.db";
const MAX_UPLOAD_SIZE: usize = 100 * 1024 * 1024; // 100MB
const MAX_QUEUE_SIZE: usize = 100;
const MAX_AUDIO_DURATION_SECS: usize = 480; // 8 minutes max to avoid ONNX runtime crashes
const SAMPLE_RATE: usize = 16000;

// Job status enum
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum JobStatus {
    Queued,
    Processing,
    Completed,
    Failed,
}

impl JobStatus {
    fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Processing => "processing",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "queued" => JobStatus::Queued,
            "processing" => JobStatus::Processing,
            "completed" => JobStatus::Completed,
            "failed" => JobStatus::Failed,
            _ => JobStatus::Failed,
        }
    }
}

// Job structure (in-memory, for queue processing)
#[derive(Debug, Clone)]
struct Job {
    id: String,
    user_uuid: String,
    message_id: String,
    status: JobStatus,
    queue_position: Option<usize>,
    audio_samples: Vec<f32>,
    params: TranscriptionParams,
    result: Option<TranscriptionResult>,
    error: Option<String>,
    created_at: i64,
}

#[derive(Debug, Clone)]
struct TranscriptionResult {
    text: String,
    tokens: Vec<TokenInfo>,
    duration: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TokenInfo {
    text: String,
    start: f32,
    end: f32,
}

// Application state
#[derive(Clone)]
struct AppState {
    jobs: Arc<RwLock<HashMap<String, Job>>>,
    queue: Arc<RwLock<VecDeque<String>>>,
    job_sender: mpsc::Sender<String>,
    db: Arc<Database>,
    model_path: PathBuf,
}

// API Response types
#[derive(Debug, Serialize)]
struct JobSubmittedResponse {
    job_id: String,
    status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    queue_position: Option<usize>,
    message: String,
    cached: bool,
}

#[derive(Debug, Serialize)]
struct JobStatusResponse {
    job_id: String,
    status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    queue_position: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_uuid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<i64>,
}

#[derive(Debug, Serialize)]
struct LookupResponse {
    found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<JobStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct LookupQuery {
    user_uuid: String,
    message_id: String,
}

#[derive(Debug, Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Debug, Serialize)]
struct VerboseTranscriptionResponse {
    task: String,
    language: String,
    duration: f64,
    text: String,
    segments: Vec<Segment>,
}

#[derive(Debug, Serialize)]
struct Segment {
    id: i32,
    seek: i32,
    start: f64,
    end: f64,
    text: String,
    tokens: Vec<i32>,
    temperature: f64,
    avg_logprob: f64,
    compression_ratio: f64,
    no_speech_prob: f64,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct TranscriptionParams {
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    temperature: Option<f64>,
    timestamp_granularities: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    version: String,
    queue_length: usize,
    db_stats: DbStatsResponse,
}

#[derive(Debug, Serialize)]
struct DbStatsResponse {
    total: usize,
    completed: usize,
    failed: usize,
    pending: usize,
}

#[derive(Debug, Serialize)]
struct QueueStatusResponse {
    queue_length: usize,
    processing: bool,
    max_queue_size: usize,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

fn create_error_response(message: &str) -> ErrorResponse {
    ErrorResponse {
        error: ErrorDetail {
            message: message.to_string(),
            r#type: "invalid_request_error".to_string(),
            param: None,
            code: None,
        },
    }
}

fn create_error_response_with_status(message: &str, status: StatusCode) -> axum::response::Response {
    (status, Json(create_error_response(message))).into_response()
}

fn now_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

// Background worker that processes jobs
async fn job_worker(
    mut receiver: mpsc::Receiver<String>,
    jobs: Arc<RwLock<HashMap<String, Job>>>,
    queue: Arc<RwLock<VecDeque<String>>>,
    db: Arc<Database>,
    model_path: PathBuf,
) {
    info!("Job worker starting, loading model...");

    // Initialize the engine once
    let engine = match ParakeetTDT::from_pretrained(model_path.to_str().unwrap_or("."), None) {
        Ok(e) => {
            info!("Model loaded successfully");
            Arc::new(Mutex::new(e))
        }
        Err(e) => {
            error!("Failed to load model: {}. Worker will not process jobs.", e);
            return;
        }
    };

    info!("Job worker ready, waiting for jobs...");

    while let Some(job_id) = receiver.recv().await {
        info!("Processing job: {}", job_id);

        // Update status in DB
        if let Err(e) = db.update_status(&job_id, "processing") {
            warn!("Failed to update job status in DB: {}", e);
        }

        // Get job data
        let job_data = {
            let mut jobs_guard = jobs.write().await;
            if let Some(job) = jobs_guard.get_mut(&job_id) {
                job.status = JobStatus::Processing;
                job.queue_position = None;
                Some((job.audio_samples.clone(), job.params.clone()))
            } else {
                None
            }
        };

        // Remove from queue and update positions
        {
            let mut queue_guard = queue.write().await;
            queue_guard.retain(|id| id != &job_id);

            // Update queue positions for remaining jobs
            let mut jobs_guard = jobs.write().await;
            for (pos, queued_id) in queue_guard.iter().enumerate() {
                if let Some(job) = jobs_guard.get_mut(queued_id) {
                    job.queue_position = Some(pos + 1);
                }
            }
        }

        if let Some((samples, _params)) = job_data {
            // Process transcription
            let result = {
                let mut engine_guard = engine.lock().await;
                engine_guard.transcribe_samples(samples, 16000, 1, Some(TimestampMode::Words))
            };

            // Update job with result
            let mut jobs_guard = jobs.write().await;
            if let Some(job) = jobs_guard.get_mut(&job_id) {
                match result {
                    Ok(transcription) => {
                        let tokens: Vec<TokenInfo> = transcription
                            .tokens
                            .iter()
                            .map(|t| TokenInfo {
                                text: t.text.clone(),
                                start: t.start,
                                end: t.end,
                            })
                            .collect();

                        let duration = transcription.tokens.last().map(|t| t.end as f64).unwrap_or(0.0);
                        let text = transcription.text.clone();

                        job.result = Some(TranscriptionResult {
                            text: text.clone(),
                            tokens,
                            duration,
                        });
                        job.status = JobStatus::Completed;

                        // Persist to database
                        if let Err(e) = db.update_completed(&job_id, &text, duration, now_timestamp()) {
                            error!("Failed to persist completed job to DB: {}", e);
                        }

                        info!("Job {} completed: {} chars", job_id, text.len());
                    }
                    Err(e) => {
                        let error_msg = format!("Transcription failed: {}", e);
                        job.error = Some(error_msg.clone());
                        job.status = JobStatus::Failed;

                        // Persist failure to database
                        if let Err(e) = db.update_failed(&job_id, &error_msg) {
                            error!("Failed to persist failed job to DB: {}", e);
                        }

                        error!("Job {} failed: {}", job_id, e);
                    }
                }
            }
        }
    }
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let queue_len = state.queue.read().await.len();
    let db_stats = state.db.get_stats().unwrap_or(db::DbStats {
        total: 0,
        completed: 0,
        failed: 0,
        pending: 0,
    });

    Json(HealthResponse {
        status: "healthy".to_string(),
        model: state.model_path.to_string_lossy().to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        queue_length: queue_len,
        db_stats: DbStatsResponse {
            total: db_stats.total,
            completed: db_stats.completed,
            failed: db_stats.failed,
            pending: db_stats.pending,
        },
    })
}

async fn queue_status(State(state): State<AppState>) -> Json<QueueStatusResponse> {
    let queue_guard = state.queue.read().await;
    let jobs_guard = state.jobs.read().await;

    let processing = jobs_guard.values().any(|j| j.status == JobStatus::Processing);

    Json(QueueStatusResponse {
        queue_length: queue_guard.len(),
        processing,
        max_queue_size: MAX_QUEUE_SIZE,
    })
}

async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "parakeet-tdt".to_string(),
                object: "model".to_string(),
                created: 1699000000,
                owned_by: "nvidia".to_string(),
            },
            ModelInfo {
                id: "whisper-1".to_string(),
                object: "model".to_string(),
                created: 1699000000,
                owned_by: "openai".to_string(),
            },
        ],
    })
}

async fn fetch_audio_from_url(url: &str) -> Result<(Vec<u8>, String)> {
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true) // For self-signed certs in dev environments
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("Failed to create HTTP client")?;

    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to fetch URL")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "HTTP error {}: {}",
            response.status().as_u16(),
            response.status().canonical_reason().unwrap_or("Unknown")
        );
    }

    // Try to get filename from Content-Disposition header or URL
    let filename = response
        .headers()
        .get("content-disposition")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| {
            s.split("filename=")
                .nth(1)
                .map(|f| f.trim_matches('"').to_string())
        })
        .or_else(|| {
            url.split('/')
                .last()
                .and_then(|s| s.split('?').next())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "audio.wav".to_string());

    let bytes = response
        .bytes()
        .await
        .context("Failed to read response body")?;

    info!("Fetched {} bytes from URL, filename: {}", bytes.len(), filename);

    Ok((bytes.to_vec(), filename))
}

async fn lookup_transcription(
    State(state): State<AppState>,
    Query(query): Query<LookupQuery>,
) -> axum::response::Response {
    // Check database for existing transcription
    match state.db.find_by_user_and_message(&query.user_uuid, &query.message_id) {
        Ok(Some(record)) => {
            Json(LookupResponse {
                found: true,
                job_id: Some(record.id),
                status: Some(JobStatus::from_str(&record.status)),
                text: record.text,
                duration: record.duration,
                error: record.error,
                created_at: Some(record.created_at),
            })
            .into_response()
        }
        Ok(None) => {
            Json(LookupResponse {
                found: false,
                job_id: None,
                status: None,
                text: None,
                duration: None,
                error: None,
                created_at: None,
            })
            .into_response()
        }
        Err(e) => {
            error!("Database lookup error: {}", e);
            create_error_response_with_status(
                "Database error",
                StatusCode::INTERNAL_SERVER_ERROR,
            )
        }
    }
}

async fn submit_transcription(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> axum::response::Response {
    let mut audio_data: Option<Vec<u8>> = None;
    let mut original_filename: Option<String> = None;
    let mut audio_url: Option<String> = None;
    let mut user_uuid: Option<String> = None;
    let mut message_id: Option<String> = None;
    let mut force_retranscribe = false;
    let mut params = TranscriptionParams::default();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                original_filename = field.file_name().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(bytes) => audio_data = Some(bytes.to_vec()),
                    Err(e) => {
                        return create_error_response(&format!("Failed to read file: {}", e))
                            .into_response()
                    }
                }
            }
            "url" => {
                if let Ok(value) = field.text().await {
                    audio_url = Some(value);
                }
            }
            "user_uuid" => {
                if let Ok(value) = field.text().await {
                    user_uuid = Some(value);
                }
            }
            "message_id" => {
                if let Ok(value) = field.text().await {
                    message_id = Some(value);
                }
            }
            "force" => {
                if let Ok(value) = field.text().await {
                    force_retranscribe = value == "true" || value == "1";
                }
            }
            "model" => {
                if let Ok(value) = field.text().await {
                    params.model = Some(value);
                }
            }
            "language" => {
                if let Ok(value) = field.text().await {
                    params.language = Some(value);
                }
            }
            "prompt" => {
                if let Ok(value) = field.text().await {
                    params.prompt = Some(value);
                }
            }
            "response_format" => {
                if let Ok(value) = field.text().await {
                    params.response_format = Some(value);
                }
            }
            "temperature" => {
                if let Ok(value) = field.text().await {
                    params.temperature = value.parse().ok();
                }
            }
            _ => {}
        }
    }

    // Validate required fields
    let user_uuid = match user_uuid {
        Some(u) if !u.is_empty() => u,
        _ => return create_error_response("user_uuid is required").into_response(),
    };

    let message_id = match message_id {
        Some(m) if !m.is_empty() => m,
        _ => return create_error_response("message_id is required").into_response(),
    };

    // Check if transcription already exists in database
    match state.db.find_by_user_and_message(&user_uuid, &message_id) {
        Ok(Some(record)) => {
            if force_retranscribe {
                // Delete existing record to force re-transcription
                info!(
                    "Force re-transcription requested for user={}, message={}",
                    user_uuid, message_id
                );
                if let Err(e) = state.db.delete_by_user_and_message(&user_uuid, &message_id) {
                    error!("Failed to delete existing transcription: {}", e);
                    return create_error_response_with_status(
                        "Database error",
                        StatusCode::INTERNAL_SERVER_ERROR,
                    );
                }
                // Continue with new transcription
            } else {
                info!(
                    "Found existing transcription for user={}, message={}",
                    user_uuid, message_id
                );
                return Json(JobSubmittedResponse {
                    job_id: record.id,
                    status: JobStatus::from_str(&record.status),
                    queue_position: None,
                    message: "Transcription already exists".to_string(),
                    cached: true,
                })
                .into_response();
            }
        }
        Ok(None) => {} // Continue with new transcription
        Err(e) => {
            error!("Database lookup error: {}", e);
            return create_error_response_with_status(
                "Database error",
                StatusCode::INTERNAL_SERVER_ERROR,
            );
        }
    }

    // Get audio data from file upload or URL
    let (audio_data, filename) = if let Some(data) = audio_data {
        let filename = original_filename.unwrap_or_else(|| "audio.wav".to_string());
        (data, filename)
    } else if let Some(url) = audio_url {
        info!("Fetching audio from URL: {}", url);
        match fetch_audio_from_url(&url).await {
            Ok((data, filename)) => (data, filename),
            Err(e) => {
                error!("Failed to fetch audio from URL: {}", e);
                return create_error_response(&format!("Failed to fetch audio from URL: {}", e))
                    .into_response();
            }
        }
    } else {
        return create_error_response("No audio file or URL provided").into_response();
    };

    // Check queue size
    {
        let queue_guard = state.queue.read().await;
        if queue_guard.len() >= MAX_QUEUE_SIZE {
            return create_error_response_with_status(
                &format!("Queue is full ({} jobs). Please try again later.", MAX_QUEUE_SIZE),
                StatusCode::TOO_MANY_REQUESTS,
            );
        }
    }

    info!(
        "Received transcription request: user={}, message={}, filename={}, size={} bytes",
        user_uuid, message_id, filename, audio_data.len()
    );

    // Convert audio to samples
    let samples = match convert_to_samples(&audio_data, &filename).await {
        Ok(s) => s,
        Err(e) => {
            return create_error_response(&format!("Failed to process audio: {}", e))
                .into_response()
        }
    };

    // Check audio duration (max 8 minutes to avoid ONNX runtime crashes)
    let duration_secs = samples.len() / SAMPLE_RATE;
    if duration_secs > MAX_AUDIO_DURATION_SECS {
        let duration_mins = duration_secs / 60;
        let max_mins = MAX_AUDIO_DURATION_SECS / 60;
        return create_error_response(&format!(
            "Audio too long: {} minutes. Maximum allowed: {} minutes. Please split the audio into smaller segments.",
            duration_mins, max_mins
        ))
        .into_response();
    }

    // Create job
    let job_id = Uuid::new_v4().to_string();
    let created_at = now_timestamp();

    // Insert into database first
    let db_record = TranscriptionRecord {
        id: job_id.clone(),
        user_uuid: user_uuid.clone(),
        message_id: message_id.clone(),
        status: "queued".to_string(),
        text: None,
        duration: None,
        error: None,
        created_at,
        completed_at: None,
    };

    if let Err(e) = state.db.insert(&db_record) {
        error!("Failed to insert job into database: {}", e);
        return create_error_response_with_status(
            "Database error",
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }

    let queue_position = {
        let mut queue_guard = state.queue.write().await;
        queue_guard.push_back(job_id.clone());
        queue_guard.len()
    };

    let job = Job {
        id: job_id.clone(),
        user_uuid: user_uuid.clone(),
        message_id: message_id.clone(),
        status: JobStatus::Queued,
        queue_position: Some(queue_position),
        audio_samples: samples,
        params,
        result: None,
        error: None,
        created_at,
    };

    // Store job in memory
    {
        let mut jobs_guard = state.jobs.write().await;
        jobs_guard.insert(job_id.clone(), job);
    }

    // Notify worker
    if let Err(e) = state.job_sender.send(job_id.clone()).await {
        error!("Failed to send job to worker: {}", e);
        return create_error_response("Internal server error").into_response();
    }

    info!("Job {} queued at position {}", job_id, queue_position);

    Json(JobSubmittedResponse {
        job_id,
        status: JobStatus::Queued,
        queue_position: Some(queue_position),
        message: format!("Job queued. Position in queue: {}", queue_position),
        cached: false,
    })
    .into_response()
}

async fn get_job_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> axum::response::Response {
    // First check in-memory jobs (for queue position info)
    {
        let jobs_guard = state.jobs.read().await;
        if let Some(job) = jobs_guard.get(&job_id) {
            return Json(JobStatusResponse {
                job_id: job.id.clone(),
                status: job.status.clone(),
                queue_position: job.queue_position,
                user_uuid: Some(job.user_uuid.clone()),
                message_id: Some(job.message_id.clone()),
                text: job.result.as_ref().map(|r| r.text.clone()),
                duration: job.result.as_ref().map(|r| r.duration),
                error: job.error.clone(),
                created_at: Some(job.created_at),
            })
            .into_response();
        }
    }

    // Fall back to database
    match state.db.find_by_id(&job_id) {
        Ok(Some(record)) => {
            Json(JobStatusResponse {
                job_id: record.id,
                status: JobStatus::from_str(&record.status),
                queue_position: None,
                user_uuid: Some(record.user_uuid),
                message_id: Some(record.message_id),
                text: record.text,
                duration: record.duration,
                error: record.error,
                created_at: Some(record.created_at),
            })
            .into_response()
        }
        Ok(None) => create_error_response_with_status(
            &format!("Job not found: {}", job_id),
            StatusCode::NOT_FOUND,
        ),
        Err(e) => {
            error!("Database error: {}", e);
            create_error_response_with_status(
                "Database error",
                StatusCode::INTERNAL_SERVER_ERROR,
            )
        }
    }
}

async fn get_job_result(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> axum::response::Response {
    // First check in-memory jobs
    {
        let jobs_guard = state.jobs.read().await;
        if let Some(job) = jobs_guard.get(&job_id) {
            return match &job.status {
                JobStatus::Completed => {
                    if let Some(result) = &job.result {
                        let response_format = job.params.response_format.as_deref().unwrap_or("json");
                        format_transcription_response(result, &job.params, response_format)
                    } else {
                        create_error_response("Result not available").into_response()
                    }
                }
                JobStatus::Failed => {
                    let error_msg = job.error.as_deref().unwrap_or("Unknown error");
                    create_error_response(error_msg).into_response()
                }
                JobStatus::Queued => {
                    let pos = job.queue_position.unwrap_or(0);
                    create_error_response_with_status(
                        &format!("Job is still queued at position {}", pos),
                        StatusCode::ACCEPTED,
                    )
                }
                JobStatus::Processing => {
                    create_error_response_with_status("Job is still processing", StatusCode::ACCEPTED)
                }
            };
        }
    }

    // Fall back to database
    match state.db.find_by_id(&job_id) {
        Ok(Some(record)) => {
            let status = JobStatus::from_str(&record.status);
            match status {
                JobStatus::Completed => {
                    if let Some(text) = record.text {
                        Json(TranscriptionResponse { text }).into_response()
                    } else {
                        create_error_response("Result not available").into_response()
                    }
                }
                JobStatus::Failed => {
                    let error_msg = record.error.as_deref().unwrap_or("Unknown error");
                    create_error_response(error_msg).into_response()
                }
                JobStatus::Queued | JobStatus::Processing => {
                    create_error_response_with_status(
                        "Job is still processing",
                        StatusCode::ACCEPTED,
                    )
                }
            }
        }
        Ok(None) => create_error_response_with_status(
            &format!("Job not found: {}", job_id),
            StatusCode::NOT_FOUND,
        ),
        Err(e) => {
            error!("Database error: {}", e);
            create_error_response_with_status(
                "Database error",
                StatusCode::INTERNAL_SERVER_ERROR,
            )
        }
    }
}

fn format_transcription_response(
    result: &TranscriptionResult,
    params: &TranscriptionParams,
    response_format: &str,
) -> axum::response::Response {
    match response_format {
        "text" => (StatusCode::OK, result.text.clone()).into_response(),
        "verbose_json" => {
            let segments: Vec<Segment> = result
                .tokens
                .iter()
                .enumerate()
                .map(|(i, token)| Segment {
                    id: i as i32,
                    seek: 0,
                    start: token.start as f64,
                    end: token.end as f64,
                    text: token.text.clone(),
                    tokens: vec![],
                    temperature: params.temperature.unwrap_or(0.0),
                    avg_logprob: 0.0,
                    compression_ratio: 0.0,
                    no_speech_prob: 0.0,
                })
                .collect();

            let verbose = VerboseTranscriptionResponse {
                task: "transcribe".to_string(),
                language: params.language.clone().unwrap_or_else(|| "en".to_string()),
                duration: result.duration,
                text: result.text.clone(),
                segments,
            };
            Json(verbose).into_response()
        }
        "srt" => {
            let mut srt = String::new();
            for (i, token) in result.tokens.iter().enumerate() {
                let start = format_srt_time(token.start);
                let end = format_srt_time(token.end);
                srt.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, start, end, token.text));
            }
            (StatusCode::OK, srt).into_response()
        }
        "vtt" => {
            let mut vtt = String::from("WEBVTT\n\n");
            for token in &result.tokens {
                let start = format_vtt_time(token.start);
                let end = format_vtt_time(token.end);
                vtt.push_str(&format!("{} --> {}\n{}\n\n", start, end, token.text));
            }
            (StatusCode::OK, vtt).into_response()
        }
        _ => Json(TranscriptionResponse {
            text: result.text.clone(),
        })
        .into_response(),
    }
}

fn format_srt_time(seconds: f32) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

fn format_vtt_time(seconds: f32) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
}

async fn convert_to_samples(data: &[u8], filename: &str) -> Result<Vec<f32>> {
    let extension = PathBuf::from(filename)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "wav".to_string());

    // For WAV files, try direct parsing first
    let samples = if extension == "wav" {
        if let Ok(samples) = parse_wav_samples(data) {
            samples
        } else {
            convert_with_symphonia(data).await?
        }
    } else {
        // For other formats, use symphonia
        convert_with_symphonia(data).await?
    };

    // Normalize audio for better transcription quality
    // This matches what pydub.effects.normalize() does in the HuggingFace space
    let samples = normalize_audio(&samples);

    Ok(samples)
}

/// Normalize audio to peak amplitude (similar to pydub.effects.normalize)
/// This improves transcription quality by ensuring consistent audio levels
fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return samples.to_vec();
    }

    // Find peak amplitude
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));

    // Avoid division by zero or amplifying silence
    if peak < 1e-6 {
        info!("Audio is silent or near-silent (peak: {}), skipping normalization", peak);
        return samples.to_vec();
    }

    // Target peak amplitude (0.95 to avoid clipping, similar to pydub's headroom)
    let target_peak = 0.95f32;
    let gain = target_peak / peak;

    // Log the normalization being applied
    let gain_db = 20.0 * gain.log10();
    info!(
        "Normalizing audio: peak={:.4}, gain={:.2}x ({:.1} dB)",
        peak, gain, gain_db
    );

    // Apply gain
    samples.iter().map(|s| s * gain).collect()
}

fn parse_wav_samples(data: &[u8]) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(data);
    let mut reader = hound::WavReader::new(cursor).context("Failed to parse WAV")?;

    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    // Convert to mono if stereo
    let samples = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let samples = if spec.sample_rate != 16000 {
        resample(&samples, spec.sample_rate, 16000)
    } else {
        samples
    };

    Ok(samples)
}

async fn convert_with_symphonia(data: &[u8]) -> Result<Vec<f32>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("Failed to probe audio format")?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("No audio track found")?;

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .context("Failed to create decoder")?;

    let track_id = track.id;
    let mut samples: Vec<f32> = Vec::new();
    let mut source_sample_rate = 16000u32;

    loop {
        match format.next_packet() {
            Ok(packet) => {
                if packet.track_id() != track_id {
                    continue;
                }

                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        source_sample_rate = decoded.spec().rate;
                        let channels = decoded.spec().channels.count();
                        let mut sample_buf =
                            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                        sample_buf.copy_interleaved_ref(decoded);

                        let src_samples = sample_buf.samples();

                        for chunk in src_samples.chunks(channels) {
                            let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                            samples.push(mono);
                        }
                    }
                    Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                    Err(_) => break,
                }
            }
            Err(_) => break,
        }
    }

    // Resample to 16kHz if needed
    let samples = if source_sample_rate != 16000 {
        resample(&samples, source_sample_rate, 16000)
    } else {
        samples
    };

    Ok(samples)
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let idx_floor = src_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(samples.len() - 1);
        let frac = src_idx - idx_floor as f64;

        let sample = if idx_floor < samples.len() {
            let s1 = samples[idx_floor] as f64;
            let s2 = samples[idx_ceil] as f64;
            (s1 + (s2 - s1) * frac) as f32
        } else {
            0.0
        };

        resampled.push(sample);
    }

    resampled
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("wazo_stt_server=info".parse().unwrap())
                .add_directive("tower_http=debug".parse().unwrap()),
        )
        .init();

    let model_path =
        PathBuf::from(env::var("MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string()));
    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());
    let db_path =
        PathBuf::from(env::var("DB_PATH").unwrap_or_else(|_| DEFAULT_DB_PATH.to_string()));

    info!("Model path: {}", model_path.display());
    info!("Database path: {}", db_path.display());

    // Ensure database directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create database directory")?;
    }

    // Initialize database
    let db = Arc::new(Database::new(&db_path).context("Failed to initialize database")?);
    info!("Database initialized");

    // Create job queue channel
    let (job_sender, job_receiver) = mpsc::channel::<String>(100);

    let jobs: Arc<RwLock<HashMap<String, Job>>> = Arc::new(RwLock::new(HashMap::new()));
    let queue: Arc<RwLock<VecDeque<String>>> = Arc::new(RwLock::new(VecDeque::new()));

    // Spawn background worker
    let worker_jobs = jobs.clone();
    let worker_queue = queue.clone();
    let worker_db = db.clone();
    let worker_model_path = model_path.clone();
    tokio::spawn(async move {
        job_worker(job_receiver, worker_jobs, worker_queue, worker_db, worker_model_path).await;
    });

    let state = AppState {
        jobs,
        queue,
        job_sender,
        db,
        model_path,
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(submit_transcription))
        .route("/v1/audio/transcriptions/lookup", get(lookup_transcription))
        .route("/v1/audio/transcriptions/status", get(queue_status))
        .route("/v1/audio/transcriptions/:job_id", get(get_job_status))
        .route("/v1/audio/transcriptions/:job_id/result", get(get_job_result))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_SIZE))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    info!("Starting server on http://{}", addr);
    info!("API endpoints:");
    info!("  POST http://{}/v1/audio/transcriptions - Submit transcription job", addr);
    info!("  GET  http://{}/v1/audio/transcriptions/lookup?user_uuid=X&message_id=Y - Lookup existing", addr);
    info!("  GET  http://{}/v1/audio/transcriptions/status - Queue status", addr);
    info!("  GET  http://{}/v1/audio/transcriptions/:job_id - Job status", addr);
    info!("  GET  http://{}/v1/audio/transcriptions/:job_id/result - Get result", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
