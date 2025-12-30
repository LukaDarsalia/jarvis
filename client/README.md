# Voice Assistant Client

A modern web-based voice assistant client that connects to a Triton Inference Server running VAD, STT, LLM, TTS, and MuseTalk models.

## Features

- **Voice-to-Voice Chat**: Speak to the assistant and hear responses
- **Avatar Video (MuseTalk)**: Lip-synced video frames generated from audio
- **Real-time Metrics**: RTF (Real-Time Factor), generation time, and audio duration display
- **Word-level Synchronization**: Visual highlighting of words as they're being spoken
- **Configurable Parameters**: Tune VAD, LLM, TTS, MuseTalk, and buffering settings from the UI

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Browser                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Voice Assistant UI                    │   │
│  │  • Voice Recording (16kHz)                               │   │
│  │  • Audio Playback (24kHz)                                │   │
│  │  • Avatar Playback (25 FPS)                              │   │
│  │  • WebSocket Communication                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     main.py                              │   │
│  │  • WebSocket endpoint (/ws)                              │   │
│  │  • REST endpoints (/health, /config)                     │   │
│  │  • Audio message routing                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  triton_client.py                        │   │
│  │  • VAD processing with state management                  │   │
│  │  • STT transcription                                     │   │
│  │  • LLM streaming generation                              │   │
│  │  • TTS streaming with word sync                          │   │
│  │  • MuseTalk video generation                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ gRPC
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Triton Inference Server                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐│
│  │   VAD    │  │   STT    │  │   LLM    │  │   TTS    │  │  MuseTalk  ││
│  │ (Silero) │  │ (NeMo)   │  │ (Kona2)  │  │ (CSM-1B) │  │  (Avatar)  ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Flow

### Voice-to-Voice Pipeline

1. **VAD (Voice Activity Detection)**
   - Detects speech onset (>200ms of speech)
   - Detects end of utterance (>1500ms of silence)
   - Accumulates audio during speech

2. **STT (Speech-to-Text)**
   - Transcribes accumulated audio
   - Returns Georgian text transcript

3. **LLM (Large Language Model)**
   - Streams response tokens
   - Uses system prompt for context

4. **TTS (Text-to-Speech)**
   - Uses 2-word lookahead for streaming
   - Generates audio frames progressively
   - Reports RTF metrics

5. **MuseTalk (Avatar)**
   - Generates video frames from TTS audio
   - Audio and video are delivered as synced 40ms frames
   - Maintains frame index continuity across turns

### TTS Streaming Protocol

The TTS model uses a 2-word lookahead system:

```
Input text: "გამარჯობა, მე ვარ ციფრული ასისტენტი"

Split into chunks:
  [0] "გამარჯობა, მე ვარ"  (first 3 words)
  [1] " ციფრული"           (word with leading space)
  [2] " ასისტენტი"         (word with leading space)
  [3] ""                   (flush marker)
  [4] ""                   (flush marker)

Generation sequence:
  When sending [0] → generates audio for nothing yet (warming up)
  When sending [1] → generates audio for nothing yet
  When sending [2] → generates audio for "გამარჯობა,"
  When sending [3] → generates audio for "მე"
  When sending [4] → generates audio for "ვარ"
  Complete        → remaining audio for "ციფრული ასისტენტი"
```

## Installation

```bash
# Navigate to client directory
cd client

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running

### Prerequisites

Make sure the Triton server is running with all models loaded:

```bash
# From project root
docker-compose up -d
```

### Start the Client

```bash
# From client directory
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Access the UI at: http://localhost:8080

## Configuration

### VAD Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speech_threshold_ms` | 200 | Minimum speech duration to consider valid |
| `silence_threshold_ms` | 1500 | Silence duration to trigger end of utterance |
| `prob_threshold` | 0.5 | Speech probability threshold |

### LLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 512 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `system_prompt` | (Georgian) | System instruction for the model |

### TTS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone_temperature` | 0.8 | Temperature for backbone model |
| `backbone_top_p` | 0.9 | Top-p for backbone sampling |
| `depth_temperature` | 0.8 | Temperature for depth decoder |
| `depth_top_p` | 0.9 | Top-p for depth decoder sampling |
| `target_sample_rate` | 24000 | Output audio sample rate |

## API Endpoints

### REST

- `GET /` - Serve web UI
- `GET /health` - Health check with model status
- `GET /config` - Get current configuration
- `POST /config` - Update configuration

### WebSocket

- `WS /ws` - Main WebSocket endpoint

#### Client → Server Messages

```json
// Recording lifecycle (pre-initialize TTS cache)
{"type": "recording_start"}
{"type": "recording_stop"}

// Stop current generation
{"type": "stop_generation"}
```

#### Server → Client Messages

```json
// Connection established
{"type": "connected", "connection_id": "...", "message": "..."}

// MuseTalk availability + idle frame
{"type": "musetalk_ready", "success": true, "idle_frame": "<base64>", "buffer_config": {...}}

// TTS cache initialized
{"type": "tts_cache_ready", "success": true}

// VAD status update
{"type": "vad_status", "status": "listening|speaking|utterance_complete"}

// STT events
{"type": "stt_start"}
{"type": "stt_complete", "text": "..."}

// LLM events
{"type": "llm_start"}
{"type": "llm_token", "token": "...", "full_text": "..."}
{"type": "llm_complete", "text": "..."}

// TTS events
{"type": "tts_start", "text": "", "video_enabled": true, "buffer_config": {...}}
{"type": "tts_complete"}

// Synced audio + video frames (40ms chunks)
{"type": "synced_av_frame", "audio": "<base64>", "frame": "<base64>", "frame_index": 0, "timestamp_ms": 0, "word": "..."}

// Video completed
{"type": "video_complete"}
```

#### Binary Messages

Audio data is sent as raw Float32 PCM at 16kHz from client to server.

## Performance Metrics

- **RTF (Real-Time Factor)**: Ratio of generation time to audio duration
  - RTF < 1.0: Faster than real-time (good)
  - RTF > 1.0: Slower than real-time (may cause delays)

## Development

### Project Structure

```
client/
├── main.py              # FastAPI backend
├── triton_client.py     # Triton gRPC client
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── static/
    ├── index.html      # Web UI
    ├── styles.css      # UI styles
    └── app.js          # Frontend JavaScript
```

### Technologies

- **Backend**: FastAPI, WebSockets, tritonclient
- **Frontend**: Vanilla JavaScript, Web Audio API
- **Communication**: WebSocket (JSON + Binary)
- **Styling**: Custom CSS with Georgian font support

## Troubleshooting

### Connection Issues

1. Check if Triton server is running: `curl http://localhost:8000/v2/health/ready`
2. Verify model status: `curl http://localhost:8000/v2/repository/index`

### Audio Issues

1. Check browser microphone permissions
2. Ensure AudioContext is not suspended (user interaction required)
3. Verify sample rates match expected values

### Performance Issues

1. Monitor RTF in the metrics panel
2. Reduce `max_new_tokens` for faster LLM responses
3. Check GPU utilization on Triton server
