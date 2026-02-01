# wazo-stt-server

An OpenAI Whisper-compatible ASR (Automatic Speech Recognition) server using NVIDIA's Parakeet model, optimized for CPU inference.

Built with Rust using [transcribe-rs](https://github.com/cjpais/transcribe-rs) for high-performance audio transcription.

## Features

- OpenAI Whisper API compatible (`/v1/audio/transcriptions`)
- NVIDIA Parakeet TDT 0.6B model (int8 quantized for CPU)
- Multiple audio format support (WAV, MP3, FLAC, OGG)
- Automatic audio conversion to required format (16kHz, mono, 16-bit)
- Docker support for easy deployment
- Multiple response formats: JSON, text, SRT, VTT, verbose_json

## Quick Start with Docker

### 1. Download the model

```bash
docker compose --profile setup run model-downloader
```

### 2. Start the server

```bash
docker compose up -d
```

The server will be available at `http://localhost:8000`.

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=parakeet"
```

#### Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | file | **Required**. Audio file to transcribe |
| `model` | string | Model to use (default: parakeet) |
| `language` | string | Language code (e.g., "en") |
| `response_format` | string | Output format: `json`, `text`, `srt`, `vtt`, `verbose_json` |
| `temperature` | number | Sampling temperature (0-1) |

#### Response (JSON)

```json
{
  "text": "The transcribed text appears here."
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/parakeet` | Path to model files |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `RUST_LOG` | `info` | Log level |

## Building from Source

### Prerequisites

- Rust 1.75+
- System dependencies: `pkg-config`, `libssl-dev`, `cmake`

### Build

```bash
cargo build --release
```

### Run

```bash
# Download model first
./scripts/download-model.sh

# Start server
MODEL_PATH=./models/parakeet cargo run --release
```

## Audio Requirements

The Parakeet model requires:
- Sample rate: 16 kHz
- Channels: Mono (1 channel)
- Format: 16-bit PCM WAV

The server automatically converts uploaded audio to the required format.

## Performance

Using int8 quantized Parakeet TDT 0.6B:
- ~5-20x real-time on modern CPUs
- ~622MB model size (encoder)

## License

MIT
