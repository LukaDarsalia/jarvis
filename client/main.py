"""
Minimal Voice Assistant FastAPI app.
- Serves static UI
- WebSocket endpoint for VAD/STT/LLM/TTS/MuseTalk pipeline
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import load_config, AppConfig
from triton_services import TritonClient
from tts_service import TTSService
from pipeline import VoiceToVoicePipeline, StreamingMetricsManager, PipelineConfig
from websocket_handler import WebSocketHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

config: Optional[AppConfig] = None
triton_client: Optional[TritonClient] = None
tts_service: Optional[TTSService] = None
pipeline: Optional[VoiceToVoicePipeline] = None
ws_handler: Optional[WebSocketHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, triton_client, tts_service, pipeline, ws_handler

    config = load_config()
    logger.info("Initializing with Triton URL: %s", config.triton_url)

    triton_client = TritonClient(
        triton_url=config.triton_url,
        vad_config=config.vad,
        llm_config=config.llm,
        tts_config=config.tts,
        musetalk_config=config.musetalk,
    )

    tts_service = TTSService(
        triton_url=config.triton_url,
        config=config.tts,
    )

    metrics_manager = StreamingMetricsManager(config.streaming)
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

    ws_handler = WebSocketHandler(
        triton_client=triton_client,
        tts_service=tts_service,
        pipeline=pipeline,
        config=config,
    )

    if triton_client.is_healthy():
        logger.info("Triton server is healthy")
    else:
        logger.warning("Triton server is not available")

    yield

    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/")
async def root():
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Voice Assistant API"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if ws_handler is None:
        await websocket.close(code=1011, reason="Server not ready")
        return

    try:
        await ws_handler.handle_connection(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        logger.exception(exc)


if __name__ == "__main__":
    import uvicorn

    os.environ.setdefault("TRITON_URL", "185.151.171.35:51954")
    uvicorn.run(app, host="0.0.0.0", port=8080)
