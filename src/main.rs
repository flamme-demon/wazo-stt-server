use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    env,
    io::Write,
    path::PathBuf,
    sync::Arc,
};
use tempfile::NamedTempFile;
use tokio::sync::Mutex;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{error, info};
use transcribe_rs::parakeet::ParakeetEngine;
use uuid::Uuid;

const DEFAULT_MODEL_PATH: &str = "/models/parakeet";
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8000";
const MAX_UPLOAD_SIZE: usize = 100 * 1024 * 1024; // 100MB

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<ParakeetEngine>>,
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
                id: "parakeet".to_string(),
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
) -> Result<impl IntoResponse, impl IntoResponse> {
    let mut audio_data: Option<Vec<u8>> = None;
    let mut original_filename: Option<String> = None;
    let mut params = TranscriptionParams::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| create_error_response(&format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                original_filename = field.file_name().map(|s| s.to_string());
                audio_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| create_error_response(&format!("Failed to read file: {}", e)))?
                        .to_vec(),
                );
            }
            "model" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| create_error_response(&format!("Failed to read model: {}", e)))?;
                params.model = Some(value);
            }
            "language" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| create_error_response(&format!("Failed to read language: {}", e)))?;
                params.language = Some(value);
            }
            "prompt" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| create_error_response(&format!("Failed to read prompt: {}", e)))?;
                params.prompt = Some(value);
            }
            "response_format" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| create_error_response(&format!("Failed to read response_format: {}", e)))?;
                params.response_format = Some(value);
            }
            "temperature" => {
                let value = field
                    .text()
                    .await
                    .map_err(|e| create_error_response(&format!("Failed to read temperature: {}", e)))?;
                params.temperature = value.parse().ok();
            }
            _ => {}
        }
    }

    let audio_data = audio_data.ok_or_else(|| create_error_response("No audio file provided"))?;
    let filename = original_filename.unwrap_or_else(|| "audio.wav".to_string());

    info!(
        "Received transcription request: filename={}, size={} bytes, model={:?}",
        filename,
        audio_data.len(),
        params.model
    );

    let temp_file = save_and_convert_audio(&audio_data, &filename)
        .await
        .map_err(|e| create_error_response(&format!("Failed to process audio: {}", e)))?;

    let temp_path = temp_file.path().to_path_buf();

    let text = {
        let mut engine = state.engine.lock().await;

        engine
            .load_model(&state.model_path)
            .map_err(|e| create_error_response(&format!("Failed to load model: {}", e)))?;

        let result = engine
            .transcribe_file(&temp_path, None)
            .map_err(|e| create_error_response(&format!("Transcription failed: {}", e)))?;

        result.text
    };

    info!("Transcription completed: {} characters", text.len());

    let response_format = params.response_format.as_deref().unwrap_or("json");

    match response_format {
        "text" => Ok((StatusCode::OK, text).into_response()),
        "verbose_json" => {
            let response = VerboseTranscriptionResponse {
                task: "transcribe".to_string(),
                language: params.language.unwrap_or_else(|| "en".to_string()),
                duration: 0.0,
                text: text.clone(),
                segments: vec![Segment {
                    id: 0,
                    seek: 0,
                    start: 0.0,
                    end: 0.0,
                    text,
                    tokens: vec![],
                    temperature: params.temperature.unwrap_or(0.0),
                    avg_logprob: 0.0,
                    compression_ratio: 0.0,
                    no_speech_prob: 0.0,
                }],
            };
            Ok(Json(response).into_response())
        }
        "srt" => {
            let srt = format!("1\n00:00:00,000 --> 00:00:00,000\n{}\n", text);
            Ok((StatusCode::OK, srt).into_response())
        }
        "vtt" => {
            let vtt = format!("WEBVTT\n\n00:00:00.000 --> 00:00:00.000\n{}\n", text);
            Ok((StatusCode::OK, vtt).into_response())
        }
        _ => Ok(Json(TranscriptionResponse { text }).into_response()),
    }
}

async fn save_and_convert_audio(data: &[u8], filename: &str) -> Result<NamedTempFile> {
    let extension = PathBuf::from(filename)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "wav".to_string());

    let is_wav = extension == "wav";

    if is_wav && is_valid_wav_format(data) {
        let mut temp_file = NamedTempFile::new().context("Failed to create temp file")?;
        temp_file.write_all(data).context("Failed to write audio data")?;
        return Ok(temp_file);
    }

    convert_to_wav(data, &extension).await
}

fn is_valid_wav_format(data: &[u8]) -> bool {
    if data.len() < 44 {
        return false;
    }

    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return false;
    }

    let channels = u16::from_le_bytes([data[22], data[23]]);
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    channels == 1 && sample_rate == 16000 && bits_per_sample == 16
}

async fn convert_to_wav(data: &[u8], _extension: &str) -> Result<NamedTempFile> {
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
    let mut samples: Vec<i16> = Vec::new();
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
                        let mut sample_buf =
                            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                        sample_buf.copy_interleaved_ref(decoded);

                        let channels = decoded.spec().channels.count();
                        let src_samples = sample_buf.samples();

                        for chunk in src_samples.chunks(channels) {
                            let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                            let sample = (mono * 32767.0).clamp(-32768.0, 32767.0) as i16;
                            samples.push(sample);
                        }
                    }
                    Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                    Err(_) => break,
                }
            }
            Err(_) => break,
        }
    }

    let samples = if source_sample_rate != 16000 {
        resample(&samples, source_sample_rate, 16000)
    } else {
        samples
    };

    let temp_file = NamedTempFile::new().context("Failed to create temp file")?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer =
        hound::WavWriter::create(temp_file.path(), spec).context("Failed to create WAV writer")?;

    for sample in samples {
        writer.write_sample(sample).context("Failed to write sample")?;
    }

    writer.finalize().context("Failed to finalize WAV")?;

    Ok(temp_file)
}

fn resample(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
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
            (s1 + (s2 - s1) * frac) as i16
        } else {
            0
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

    let model_path = PathBuf::from(
        env::var("MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string()),
    );
    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());

    info!("Initializing Parakeet engine...");
    let engine = ParakeetEngine::new();

    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        model_path: model_path.clone(),
    };

    info!("Model path: {}", model_path.display());

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
