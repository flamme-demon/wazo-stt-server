use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
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
use tracing::{error, info};
use uuid::Uuid;

const DEFAULT_MODEL_PATH: &str = "/models/parakeet";
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8000";
const MAX_UPLOAD_SIZE: usize = 100 * 1024 * 1024; // 100MB
const MAX_QUEUE_SIZE: usize = 100;

// Job status enum
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum JobStatus {
    Queued,
    Processing,
    Completed,
    Failed,
}

// Job structure
#[derive(Debug, Clone)]
struct Job {
    id: String,
    status: JobStatus,
    queue_position: Option<usize>,
    audio_samples: Vec<f32>,
    params: TranscriptionParams,
    result: Option<TranscriptionResult>,
    error: Option<String>,
    created_at: u64,
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
    model_path: PathBuf,
}

// API Response types
#[derive(Debug, Serialize)]
struct JobSubmittedResponse {
    job_id: String,
    status: JobStatus,
    queue_position: usize,
    message: String,
}

#[derive(Debug, Serialize)]
struct JobStatusResponse {
    job_id: String,
    status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    queue_position: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<u64>,
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

// Background worker that processes jobs
async fn job_worker(
    mut receiver: mpsc::Receiver<String>,
    jobs: Arc<RwLock<HashMap<String, Job>>>,
    queue: Arc<RwLock<VecDeque<String>>>,
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

        if let Some((samples, params)) = job_data {
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

                        job.result = Some(TranscriptionResult {
                            text: transcription.text.clone(),
                            tokens,
                            duration,
                        });
                        job.status = JobStatus::Completed;
                        info!("Job {} completed: {} chars", job_id, transcription.text.len());
                    }
                    Err(e) => {
                        job.error = Some(format!("Transcription failed: {}", e));
                        job.status = JobStatus::Failed;
                        error!("Job {} failed: {}", job_id, e);
                    }
                }
            }
        }
    }
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let queue_len = state.queue.read().await.len();
    Json(HealthResponse {
        status: "healthy".to_string(),
        model: state.model_path.to_string_lossy().to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        queue_length: queue_len,
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

async fn submit_transcription(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> axum::response::Response {
    let mut audio_data: Option<Vec<u8>> = None;
    let mut original_filename: Option<String> = None;
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

    let audio_data = match audio_data {
        Some(data) => data,
        None => return create_error_response("No audio file provided").into_response(),
    };
    let filename = original_filename.unwrap_or_else(|| "audio.wav".to_string());

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
        "Received transcription request: filename={}, size={} bytes",
        filename,
        audio_data.len()
    );

    // Convert audio to samples
    let samples = match convert_to_samples(&audio_data, &filename).await {
        Ok(s) => s,
        Err(e) => {
            return create_error_response(&format!("Failed to process audio: {}", e))
                .into_response()
        }
    };

    // Create job
    let job_id = Uuid::new_v4().to_string();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let queue_position = {
        let mut queue_guard = state.queue.write().await;
        queue_guard.push_back(job_id.clone());
        queue_guard.len()
    };

    let job = Job {
        id: job_id.clone(),
        status: JobStatus::Queued,
        queue_position: Some(queue_position),
        audio_samples: samples,
        params,
        result: None,
        error: None,
        created_at,
    };

    // Store job
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
        queue_position,
        message: format!("Job queued. Position in queue: {}", queue_position),
    })
    .into_response()
}

async fn get_job_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> axum::response::Response {
    let jobs_guard = state.jobs.read().await;

    match jobs_guard.get(&job_id) {
        Some(job) => {
            let response = JobStatusResponse {
                job_id: job.id.clone(),
                status: job.status.clone(),
                queue_position: job.queue_position,
                text: job.result.as_ref().map(|r| r.text.clone()),
                duration: job.result.as_ref().map(|r| r.duration),
                error: job.error.clone(),
                created_at: Some(job.created_at),
            };
            Json(response).into_response()
        }
        None => create_error_response_with_status(
            &format!("Job not found: {}", job_id),
            StatusCode::NOT_FOUND,
        ),
    }
}

async fn get_job_result(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> axum::response::Response {
    let jobs_guard = state.jobs.read().await;

    match jobs_guard.get(&job_id) {
        Some(job) => match &job.status {
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
        },
        None => create_error_response_with_status(
            &format!("Job not found: {}", job_id),
            StatusCode::NOT_FOUND,
        ),
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
    if extension == "wav" {
        if let Ok(samples) = parse_wav_samples(data) {
            return Ok(samples);
        }
    }

    // For other formats, use symphonia
    convert_with_symphonia(data).await
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

    info!("Model path: {}", model_path.display());

    // Create job queue channel
    let (job_sender, job_receiver) = mpsc::channel::<String>(100);

    let jobs: Arc<RwLock<HashMap<String, Job>>> = Arc::new(RwLock::new(HashMap::new()));
    let queue: Arc<RwLock<VecDeque<String>>> = Arc::new(RwLock::new(VecDeque::new()));

    // Spawn background worker
    let worker_jobs = jobs.clone();
    let worker_queue = queue.clone();
    let worker_model_path = model_path.clone();
    tokio::spawn(async move {
        job_worker(job_receiver, worker_jobs, worker_queue, worker_model_path).await;
    });

    let state = AppState {
        jobs,
        queue,
        job_sender,
        model_path,
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(submit_transcription))
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
    info!("  GET  http://{}/v1/audio/transcriptions/status - Queue status", addr);
    info!("  GET  http://{}/v1/audio/transcriptions/:job_id - Job status", addr);
    info!("  GET  http://{}/v1/audio/transcriptions/:job_id/result - Get result", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
