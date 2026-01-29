"""
Voice Assistant FastAPI Application.

A WebSocket-based voice assistant with:
- Voice Activity Detection (VAD)
- Speech-to-Text (STT)
- Large Language Model (LLM) streaming
- Text-to-Speech (TTS) with 2-word lookahead
- MuseTalk avatar video generation
- Synchronized audio/video streaming

Architecture:
    User (WebSocket) <-> FastAPI <-> Triton (gRPC)
                                       |
                                       v
                            VAD, STT, LLM, TTS, MuseTalk
"""

from __future__ import annotations

import logging
import os
import sys
import re
import tempfile
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import load_config, AppConfig
from triton_services import TritonClient
from tts_service import TTSService
from pipeline import VoiceToVoicePipeline, StreamingMetricsManager, PipelineConfig
from websocket_handler import WebSocketHandler

import numpy as np
import cv2

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception!", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = global_exception_handler


# ============================================================================
# Global State
# ============================================================================

# Loaded during lifespan
config: Optional[AppConfig] = None
triton_client: Optional[TritonClient] = None
tts_service: Optional[TTSService] = None
pipeline: Optional[VoiceToVoicePipeline] = None
ws_handler: Optional[WebSocketHandler] = None


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, triton_client, tts_service, pipeline, ws_handler

    # Load configuration
    config = load_config()
    logger.info(f"Initializing with Triton URL: {config.triton_url}")

    # Create Triton client
    triton_client = TritonClient(
        triton_url=config.triton_url,
        vad_config=config.vad,
        llm_config=config.llm,
        tts_config=config.tts,
        musetalk_config=config.musetalk,
    )

    # Create TTS service
    tts_service = TTSService(
        triton_url=config.triton_url,
        config=config.tts,
    )

    # Create metrics manager
    metrics_manager = StreamingMetricsManager(config.streaming)

    # Create pipeline
    pipeline_config = PipelineConfig(
        tts_config=config.tts,
        musetalk_config=config.musetalk,
        streaming_config=config.streaming,
    )
    pipeline = VoiceToVoicePipeline(
        tts_service=tts_service,
        musetalk_service=triton_client.musetalk,
        metrics_manager=metrics_manager,
        config=pipeline_config,
    )

    # Create WebSocket handler
    ws_handler = WebSocketHandler(
        triton_client=triton_client,
        tts_service=tts_service,
        pipeline=pipeline,
        config=config,
    )

    # Health check
    if triton_client.is_healthy():
        logger.info("Triton server is healthy")
        logger.info(f"Models status: {triton_client.check_models_ready()}")
    else:
        logger.warning("Triton server is not available - client will retry on requests")

    yield

    logger.info("Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Voice Assistant API",
    description="WebSocket-based voice assistant with VAD, STT, LLM, TTS, and avatar generation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

AVATAR_ROOT = os.environ.get(
    "AVATAR_ROOT",
    "/local_models/musetalk_model/testing_avatar_creation/v15/avatars",
)
AVATAR_RESULT_DIR = os.environ.get(
    "AVATAR_RESULT_DIR",
    "/local_models/musetalk_model/testing_avatar_creation",
)
AVATAR_PYTHON = os.environ.get("AVATAR_PYTHON", "/opt/avatar_venv/bin/python")
AVATAR_SCRIPT = os.environ.get("AVATAR_CREATE_SCRIPT", "/app/musetalk/create_avatar.py")
AVATAR_MODEL_ROOT = os.environ.get("AVATAR_MODEL_ROOT", "/local_models/musetalk_model")
AVATAR_DEFAULT_VERSION = os.environ.get("AVATAR_VERSION", "v15")
AVATAR_DEVICE = os.environ.get("AVATAR_DEVICE", "cuda")
AVATAR_MAX_SIDE = os.environ.get("AVATAR_MAX_SIDE", "0")


def _sanitize_avatar_id(raw: str) -> Optional[str]:
    if not raw:
        return None
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")
    return name[:48] if name else None


def _list_avatars() -> List[str]:
    if not os.path.isdir(AVATAR_ROOT):
        return []
    items: List[str] = []
    for entry in os.listdir(AVATAR_ROOT):
        full = os.path.join(AVATAR_ROOT, entry)
        if os.path.isdir(full):
            items.append(entry)
    return sorted(items)


def _write_image_video(
    image_bytes: bytes,
    output_path: str,
    fps: int = 25,
    duration_s: float = 1.0,
) -> None:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image data")
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions")
    frame_count = max(1, int(round(fps * duration_s)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter for avatar video")
    for _ in range(frame_count):
        writer.write(img)
    writer.release()


# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Serve the main page."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Voice Assistant API - WebSocket endpoint at /ws"}


@app.get("/health")
async def health():
    """Check server and model health."""
    if triton_client is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Client not initialized"},
        )

    server_healthy = triton_client.is_healthy()
    models_status = triton_client.check_models_ready()

    return {
        "status": "healthy" if server_healthy else "degraded",
        "triton_server": server_healthy,
        "models": models_status,
    }


@app.get("/config")
async def get_config():
    """Get current configuration."""
    if config is None or triton_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")

    return {
        "vad": {
            "speech_threshold_ms": config.vad.speech_threshold_ms,
            "silence_threshold_ms": config.vad.silence_threshold_ms,
            "early_silence_threshold_ms": config.vad.early_silence_threshold_ms,
            "enable_speculative": config.vad.enable_speculative,
            "prob_threshold": config.vad.prob_threshold,
        },
        "llm": {
            "max_new_tokens": config.llm.max_new_tokens,
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "system_prompt": config.llm.system_prompt,
        },
        "tts": {
            "backbone_temperature": config.tts.backbone_temperature,
            "backbone_top_p": config.tts.backbone_top_p,
            "depth_temperature": config.tts.depth_temperature,
            "depth_top_p": config.tts.depth_top_p,
            "sample_rate": config.tts.sample_rate,
            "voice_id": config.tts.voice_id,
        },
        "musetalk": {
            "batch_size": config.musetalk.batch_size,
            "lookahead_chunks": config.musetalk.lookahead_chunks,
            "avatar_id": config.musetalk.avatar_id,
        },
    }


@app.get("/avatars")
async def list_avatars():
    """List available avatar IDs."""
    return {"avatars": _list_avatars()}


@app.post("/avatar/create")
async def create_avatar(
    image: UploadFile = File(...),
    avatar_id: str = Form(...),
    force_recreate: bool = Form(False),
    fps: int = Form(25),
    version: str = Form(AVATAR_DEFAULT_VERSION),
    duration_s: float = Form(1.0),
):
    """Create a MuseTalk avatar from an uploaded image."""
    safe_avatar_id = _sanitize_avatar_id(avatar_id)
    if not safe_avatar_id:
        raise HTTPException(status_code=400, detail="Invalid avatar_id")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    os.makedirs(AVATAR_RESULT_DIR, exist_ok=True)
    if not os.path.exists(AVATAR_PYTHON):
        raise HTTPException(status_code=500, detail="Avatar venv Python not found")
    if not os.path.exists(AVATAR_SCRIPT):
        raise HTTPException(status_code=500, detail="Avatar creation script not found")

    temp_dir = tempfile.mkdtemp(prefix="avatar_")
    image_path = os.path.join(temp_dir, f"{safe_avatar_id}.png")
    video_path = os.path.join(temp_dir, f"{safe_avatar_id}.mp4")

    try:
        with open(image_path, "wb") as f:
            f.write(data)

        _write_image_video(data, video_path, fps=fps, duration_s=duration_s)

        cmd = [
            AVATAR_PYTHON,
            AVATAR_SCRIPT,
            "--video_path",
            video_path,
            "--result_dir",
            AVATAR_RESULT_DIR,
            "--version",
            version,
            "--model_root",
            AVATAR_MODEL_ROOT,
            "--device",
            AVATAR_DEVICE,
            "--max_side",
            AVATAR_MAX_SIDE,
            "--fps",
            str(fps),
            "--batch_size",
            "8",
            "--avatar_id",
            safe_avatar_id,
            "--force_recreate" if force_recreate else "",
        ]
        cmd = [c for c in cmd if c]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", errors="ignore")
            raise HTTPException(status_code=500, detail=detail or "Avatar creation failed")

        return {"avatar_id": safe_avatar_id}
    finally:
        try:
            for path in (image_path, video_path):
                if os.path.exists(path):
                    os.unlink(path)
            os.rmdir(temp_dir)
        except OSError:
            pass


@app.post("/config")
async def update_config(new_config: dict):
    """Update configuration."""
    if config is None or triton_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")

    if "vad" in new_config:
        vad = new_config["vad"]
        if "speech_threshold_ms" in vad:
            config.vad.speech_threshold_ms = float(vad["speech_threshold_ms"])
        if "silence_threshold_ms" in vad:
            config.vad.silence_threshold_ms = float(vad["silence_threshold_ms"])
        if "early_silence_threshold_ms" in vad:
            config.vad.early_silence_threshold_ms = float(vad["early_silence_threshold_ms"])
        if "enable_speculative" in vad:
            config.vad.enable_speculative = bool(vad["enable_speculative"])
        if "prob_threshold" in vad:
            config.vad.prob_threshold = float(vad["prob_threshold"])

    if "llm" in new_config:
        llm = new_config["llm"]
        if "max_new_tokens" in llm:
            config.llm.max_new_tokens = int(llm["max_new_tokens"])
        if "temperature" in llm:
            config.llm.temperature = float(llm["temperature"])
        if "top_p" in llm:
            config.llm.top_p = float(llm["top_p"])
        if "system_prompt" in llm:
            config.llm.system_prompt = str(llm["system_prompt"])
            logger.info(f"[CONFIG] Updated system_prompt to: {config.llm.system_prompt[:50]}...")

    if "tts" in new_config:
        tts = new_config["tts"]
        if "backbone_temperature" in tts:
            config.tts.backbone_temperature = float(tts["backbone_temperature"])
        if "backbone_top_p" in tts:
            config.tts.backbone_top_p = float(tts["backbone_top_p"])
        if "depth_temperature" in tts:
            config.tts.depth_temperature = float(tts["depth_temperature"])
        if "depth_top_p" in tts:
            config.tts.depth_top_p = float(tts["depth_top_p"])
        if "sample_rate" in tts:
            config.tts.sample_rate = int(tts["sample_rate"])
        if "voice_id" in tts:
            config.tts.voice_id = str(tts["voice_id"])

    if "musetalk" in new_config:
        mt_cfg = new_config["musetalk"]
        if "batch_size" in mt_cfg:
            try:
                config.musetalk.batch_size = max(1, min(32, int(mt_cfg["batch_size"])))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid batch_size: {mt_cfg['batch_size']}")
        if "lookahead_chunks" in mt_cfg:
            try:
                config.musetalk.lookahead_chunks = max(0, int(mt_cfg["lookahead_chunks"]))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid lookahead_chunks: {mt_cfg['lookahead_chunks']}")
        if "avatar_id" in mt_cfg:
            config.musetalk.avatar_id = str(mt_cfg["avatar_id"])

    return await get_config()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice assistant."""
    if ws_handler is None:
        await websocket.close(code=1011, reason="Server not ready")
        return

    try:
        await ws_handler.handle_connection(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        import traceback
        logger.error(traceback.format_exc())


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    os.environ.setdefault("TRITON_URL", "localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8000)
