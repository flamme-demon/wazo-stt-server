use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use serde::{Deserialize, Serialize};
use std::{env, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

const DEFAULT_MODEL_PATH: &str = "/models/parakeet";
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8000";
const MAX_UPLOAD_SIZE: usize = 100 * 1024 * 1024; // 100MB

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Option<ParakeetTDT>>>,
    model_path: PathBuf,
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

#[derive(Debug, Deserialize, Default)]
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

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        model: state.model_path.to_string_lossy().to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
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

async fn transcribe(
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

    info!(
        "Received transcription request: filename={}, size={} bytes, model={:?}",
        filename,
        audio_data.len(),
        params.model
    );

    // Convert audio to required format
    let samples = match convert_to_samples(&audio_data, &filename).await {
        Ok(s) => s,
        Err(e) => {
            return create_error_response(&format!("Failed to process audio: {}", e))
                .into_response()
        }
    };

    // Transcribe
    let result = {
        let mut engine_guard = state.engine.lock().await;

        // Initialize engine if not already done
        if engine_guard.is_none() {
            info!("Loading Parakeet TDT model from {:?}", state.model_path);
            match ParakeetTDT::from_pretrained(state.model_path.to_str().unwrap_or("."), None) {
                Ok(engine) => *engine_guard = Some(engine),
                Err(e) => {
                    return create_error_response(&format!("Failed to load model: {}", e))
                        .into_response()
                }
            }
        }

        let engine = engine_guard.as_mut().unwrap();

        match engine.transcribe_samples(samples, 16000, 1, Some(TimestampMode::Words)) {
            Ok(r) => r,
            Err(e) => {
                return create_error_response(&format!("Transcription failed: {}", e))
                    .into_response()
            }
        }
    };

    let text = result.text.clone();
    info!("Transcription completed: {} characters", text.len());

    let response_format = params.response_format.as_deref().unwrap_or("json");

    match response_format {
        "text" => (StatusCode::OK, text).into_response(),
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
                language: params.language.unwrap_or_else(|| "en".to_string()),
                duration: result.tokens.last().map(|t| t.end as f64).unwrap_or(0.0),
                text: result.text,
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
        _ => Json(TranscriptionResponse { text }).into_response(),
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

    let state = AppState {
        engine: Arc::new(Mutex::new(None)),
        model_path,
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(transcribe))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_SIZE))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    info!("Starting server on http://{}", addr);
    info!("API endpoint: POST http://{}/v1/audio/transcriptions", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
