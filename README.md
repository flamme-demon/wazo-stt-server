# Wazo STT Server

Serveur de transcription automatique (ASR) compatible OpenAI Whisper utilisant le modèle NVIDIA Parakeet via `onnx_asr`, optimisé pour l'inférence CPU.

## Fonctionnalités

- API compatible OpenAI Whisper (`/v1/audio/transcriptions`)
- Modèle NVIDIA Parakeet TDT 0.6B via `onnx_asr`
- **File d'attente asynchrone** avec suivi de position
- **Persistance SQLite** des transcriptions
- **Lookup par user_uuid/message_id** pour éviter les re-transcriptions
- **Fetch audio depuis URL** (Wazo, etc.)
- Support multi-format audio (WAV, MP3, FLAC, OGG, etc.)
- Normalisation audio automatique
- Conversion automatique (16kHz, mono)
- Formats de sortie : JSON, text, SRT, VTT, verbose_json

## Démarrage rapide

### Avec Docker (recommandé)

```bash
# Construire et démarrer
docker compose up -d --build
```

Le modèle est téléchargé automatiquement au premier lancement.
Le serveur est disponible sur `http://localhost:8000`.

### Sans Docker

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
python main.py
```

## Utilisation

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

### Forcer la re-transcription

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "url=https://example.com/audio.wav" \
  -F "user_uuid=abc-123" \
  -F "message_id=msg-456" \
  -F "force=true"
```

### Vérifier si une transcription existe

```bash
curl "http://localhost:8000/v1/audio/transcriptions/lookup?user_uuid=abc-123&message_id=msg-456"
```

### Récupérer le statut d'un job

```bash
curl http://localhost:8000/v1/audio/transcriptions/{job_id}
```

### Récupérer le résultat

```bash
curl http://localhost:8000/v1/audio/transcriptions/{job_id}/result
```

## Documentation API

Voir [API.md](API.md) pour la documentation complète de l'API.

## Configuration

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MODEL_NAME` | `nemo-parakeet-tdt-0.6b-v3` | Nom du modèle onnx_asr |
| `DB_PATH` | `/data/transcriptions.db` | Chemin de la base SQLite |
| `HOST` | `0.0.0.0` | Adresse de bind |
| `PORT` | `8000` | Port du serveur |
| `MAX_QUEUE_SIZE` | `100` | Taille max de la file d'attente |
| `MAX_AUDIO_DURATION_SECS` | `480` | Durée max audio (8 min) |
| `TEXT_RETENTION_DAYS` | `365` | Durée de rétention du texte (jours) |

## Volumes Docker

| Volume | Chemin | Description |
|--------|--------|-------------|
| `wazo-stt-data` | `/data` | Base de données SQLite |
| `wazo-stt-model-cache` | `/root/.cache` | Cache du modèle onnx_asr |

## Performance

- Transcription ~5-20x temps réel sur CPU moderne
- ~2GB RAM utilisée
- Le modèle est téléchargé et mis en cache au premier lancement

## Stack technique

- **Python 3.11** + FastAPI
- **onnx_asr** pour la transcription (NVIDIA Parakeet)
- **pydub** pour le traitement audio
- **SQLite** pour la persistance
- **Docker** pour le déploiement

## License

MIT
