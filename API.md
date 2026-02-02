# Wazo STT Server - Documentation API

Base URL: `http://localhost:8000`

## Authentification

Aucune authentification requise côté serveur STT. L'authentification Wazo est gérée via le token dans l'URL de l'audio.

---

## Endpoints

### 1. Soumettre une transcription

Soumet un fichier audio ou une URL pour transcription.

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

#### Paramètres (form-data)

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `user_uuid` | string | **Oui** | UUID de l'utilisateur Wazo |
| `message_id` | string | **Oui** | ID du message vocal |
| `file` | file | Non* | Fichier audio à transcrire |
| `url` | string | Non* | URL du fichier audio (avec token si nécessaire) |
| `model` | string | Non | Modèle à utiliser (défaut: parakeet) |
| `language` | string | Non | Code langue (ex: "fr", "en") |
| `response_format` | string | Non | Format de sortie: `json`, `text`, `srt`, `vtt`, `verbose_json` |

*\* `file` ou `url` doit être fourni*

#### Exemple avec URL (recommandé pour Wazo)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "url=https://wazo.example.com/api/calld/1.0/users/me/voicemails/messages/1712847495-00000002/recording?token=xxx" \
  -F "user_uuid=550e8400-e29b-41d4-a716-446655440000" \
  -F "message_id=1712847495-00000002"
```

#### Exemple avec fichier

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "user_uuid=550e8400-e29b-41d4-a716-446655440000" \
  -F "message_id=1712847495-00000002"
```

#### Réponse - Nouveau job

```json
{
  "job_id": "19a0cf67-2b65-4dd0-a046-fc9e3a4f8624",
  "status": "queued",
  "queue_position": 1,
  "message": "Job queued. Position in queue: 1",
  "cached": false
}
```

#### Réponse - Transcription existante (cache)

```json
{
  "job_id": "19a0cf67-2b65-4dd0-a046-fc9e3a4f8624",
  "status": "completed",
  "queue_position": null,
  "message": "Transcription already exists",
  "cached": true
}
```

---

### 2. Lookup - Vérifier si une transcription existe

Vérifie si une transcription existe déjà pour un couple user_uuid/message_id.

```
GET /v1/audio/transcriptions/lookup?user_uuid={uuid}&message_id={id}
```

#### Paramètres (query)

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `user_uuid` | string | **Oui** | UUID de l'utilisateur Wazo |
| `message_id` | string | **Oui** | ID du message vocal |

#### Exemple

```bash
curl "http://localhost:8000/v1/audio/transcriptions/lookup?user_uuid=550e8400-e29b-41d4-a716-446655440000&message_id=1712847495-00000002"
```

#### Réponse - Trouvé

```json
{
  "found": true,
  "job_id": "19a0cf67-2b65-4dd0-a046-fc9e3a4f8624",
  "status": "completed",
  "text": "Bonjour, ceci est un message vocal...",
  "duration": 12.5,
  "created_at": 1770030887
}
```

#### Réponse - Non trouvé

```json
{
  "found": false
}
```

---

### 3. Statut d'un job

Récupère le statut et le résultat d'un job de transcription.

```
GET /v1/audio/transcriptions/{job_id}
```

#### Exemple

```bash
curl http://localhost:8000/v1/audio/transcriptions/19a0cf67-2b65-4dd0-a046-fc9e3a4f8624
```

#### Réponse - En attente

```json
{
  "job_id": "19a0cf67-2b65-4dd0-a046-fc9e3a4f8624",
  "status": "queued",
  "queue_position": 2,
  "user_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "1712847495-00000002",
  "created_at": 1770030887
}
```

#### Réponse - En cours

```json
{
  "job_id": "19a0cf67-2b65-4dd0-a046-fc9e3a4f8624",
  "status": "processing",
  "user_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "1712847495-00000002",
  "created_at": 1770030887
}
```

#### Réponse - Terminé

```json
{
  "job_id": "19a0cf67-2b65-4dd0-a046-fc9e3a4f8624",
  "status": "completed",
  "user_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "1712847495-00000002",
  "text": "Bonjour, ceci est un message vocal transcrit automatiquement.",
  "duration": 12.5,
  "created_at": 1770030887
}
```

#### Réponse - Erreur

```json
{
  "job_id": "19a0cf67-2b65-4dd0-a046-fc9e3a4f8624",
  "status": "failed",
  "user_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "1712847495-00000002",
  "error": "Transcription failed: audio too short",
  "created_at": 1770030887
}
```

---

### 4. Résultat formaté

Récupère le résultat dans le format demandé lors de la soumission.

```
GET /v1/audio/transcriptions/{job_id}/result
```

#### Réponses selon le format

**JSON (défaut)**
```json
{
  "text": "Bonjour, ceci est un message vocal."
}
```

**text**
```
Bonjour, ceci est un message vocal.
```

**srt**
```
1
00:00:00,000 --> 00:00:02,500
Bonjour

2
00:00:02,500 --> 00:00:05,000
ceci est un message vocal
```

**vtt**
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Bonjour

00:00:02.500 --> 00:00:05.000
ceci est un message vocal
```

---

### 5. Statut de la file d'attente

```
GET /v1/audio/transcriptions/status
```

#### Réponse

```json
{
  "queue_length": 3,
  "processing": true,
  "max_queue_size": 100
}
```

---

### 6. Health check

```
GET /health
```

#### Réponse

```json
{
  "status": "healthy",
  "model": "/models/parakeet",
  "version": "0.1.0",
  "queue_length": 0,
  "db_stats": {
    "total": 150,
    "completed": 145,
    "failed": 3,
    "pending": 2
  }
}
```

---

### 7. Liste des modèles

```
GET /v1/models
```

#### Réponse

```json
{
  "object": "list",
  "data": [
    {
      "id": "parakeet-tdt",
      "object": "model",
      "created": 1699000000,
      "owned_by": "nvidia"
    }
  ]
}
```

---

## Workflow recommandé pour le plugin client

```
┌─────────────────────────────────────────────────────────┐
│                    Plugin Client                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 1. GET /lookup?user_uuid=X&message_id=Y                 │
│                                                         │
│    found=true  ──────► Afficher le texte               │
│    found=false ──────► Continuer étape 2               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 2. POST /v1/audio/transcriptions                        │
│    - url: URL Wazo du recording (avec token)            │
│    - user_uuid: UUID utilisateur                        │
│    - message_id: ID du message vocal                    │
│                                                         │
│    cached=true  ──────► Afficher le texte              │
│    cached=false ──────► Continuer étape 3              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Poll GET /v1/audio/transcriptions/{job_id}           │
│    (toutes les 1-2 secondes)                            │
│                                                         │
│    status=queued     ──────► Afficher position         │
│    status=processing ──────► Afficher "En cours..."    │
│    status=completed  ──────► Afficher le texte         │
│    status=failed     ──────► Afficher l'erreur         │
└─────────────────────────────────────────────────────────┘
```

---

## Codes d'erreur HTTP

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 202 | Job en cours (pas encore terminé) |
| 400 | Requête invalide (paramètre manquant) |
| 404 | Job non trouvé |
| 429 | File d'attente pleine |
| 500 | Erreur serveur |

---

## Format des erreurs

```json
{
  "error": {
    "message": "user_uuid is required",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

---

## Statuts des jobs

| Statut | Description |
|--------|-------------|
| `queued` | En attente dans la file |
| `processing` | Transcription en cours |
| `completed` | Terminé avec succès |
| `failed` | Échec de la transcription |

---

## Notes importantes

1. **Unicité** : La combinaison `(user_uuid, message_id)` est unique. Soumettre le même couple retournera la transcription existante.

2. **Persistance** : Les transcriptions sont stockées en base SQLite et survivent aux redémarrages du serveur.

3. **Timeout URL** : Le fetch des URLs a un timeout de 60 secondes.

4. **Certificats** : Les certificats auto-signés sont acceptés pour les environnements de développement.

5. **Formats audio** : WAV, MP3, FLAC, OGG sont supportés. La conversion en 16kHz mono est automatique.
