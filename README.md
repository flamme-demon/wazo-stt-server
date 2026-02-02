# Wazo STT Server

Serveur de transcription automatique (ASR) compatible OpenAI Whisper utilisant le modèle NVIDIA Parakeet, optimisé pour l'inférence CPU.

## Fonctionnalités

- API compatible OpenAI Whisper (`/v1/audio/transcriptions`)
- Modèle NVIDIA Parakeet TDT 0.6B (quantifié int8 pour CPU)
- **File d'attente asynchrone** avec suivi de position
- **Persistance SQLite** des transcriptions
- **Lookup par user_uuid/message_id** pour éviter les re-transcriptions
- **Fetch audio depuis URL** (Wazo, etc.)
- Support multi-format audio (WAV, MP3, FLAC, OGG)
- Conversion automatique (16kHz, mono)
- Formats de sortie : JSON, text, SRT, VTT, verbose_json

## Démarrage rapide

### 1. Télécharger le modèle

```bash
docker compose --profile setup run model-downloader
```

### 2. Démarrer le serveur

```bash
docker compose up -d
```

Le serveur est disponible sur `http://localhost:8000`.

## Utilisation basique

### Transcrire un fichier audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "user_uuid=abc-123" \
  -F "message_id=msg-456"
```

### Transcrire depuis une URL

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "url=https://example.com/audio.wav?token=xxx" \
  -F "user_uuid=abc-123" \
  -F "message_id=msg-456"
```

### Vérifier si une transcription existe

```bash
curl "http://localhost:8000/v1/audio/transcriptions/lookup?user_uuid=abc-123&message_id=msg-456"
```

### Récupérer le statut d'un job

```bash
curl http://localhost:8000/v1/audio/transcriptions/{job_id}
```

## Documentation API

Voir [API.md](API.md) pour la documentation complète de l'API.

## Configuration

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MODEL_PATH` | `/models/parakeet` | Chemin vers les fichiers du modèle |
| `DB_PATH` | `/data/transcriptions.db` | Chemin de la base SQLite |
| `HOST` | `0.0.0.0` | Adresse de bind |
| `PORT` | `8000` | Port du serveur |
| `RUST_LOG` | `info` | Niveau de log |

## Volumes Docker

| Volume | Chemin | Description |
|--------|--------|-------------|
| `wazo-stt-models` | `/models` | Fichiers du modèle Parakeet |
| `wazo-stt-data` | `/data` | Base de données SQLite |

## Performance

Avec Parakeet TDT 0.6B int8 :
- ~5-20x temps réel sur CPU moderne
- ~622MB taille du modèle (encoder)
- ~2GB RAM utilisée

## Compilation depuis les sources

### Prérequis

- Rust 1.85+
- Dépendances : `pkg-config`, `libssl-dev`, `cmake`

### Build

```bash
cargo build --release
```

### Run

```bash
MODEL_PATH=./models/parakeet DB_PATH=./data/transcriptions.db cargo run --release
```

## License

MIT
