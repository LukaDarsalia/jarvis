#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
import wave
from queue import Queue

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from pipeline import AVPipeline, StreamingMetricsManager, PipelineConfig
from streaming_chunker import StreamingTTSChunker, split_text_for_streaming
from tts_service import TTSService
os.environ.setdefault("TRITON_URL", "185.151.171.35:54757")


class DummyMuseTalk:
    def generate_frames(self, *args, **kwargs):
        return iter(())


def iter_tokens(text: str, mode: str, size: int):
    if mode == "full":
        yield text
        return
    if mode == "words":
        for word in text.split():
            yield word + " "
        return
    if mode == "chars":
        for i in range(0, len(text), size):
            yield text[i:i + size]
        return
    raise ValueError(f"Unknown token mode: {mode}")


def build_chunks(text: str, mode: str, size: int):
    if mode == "full":
        return split_text_for_streaming(text)
    chunker = StreamingTTSChunker()
    chunks = []
    for token in iter_tokens(text, mode, size):
        chunks.extend(chunker.push_token(token))
    chunks.extend(chunker.finalize())
    return chunks


def write_wav(path: str, audio: np.ndarray, sample_rate: int):
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


async def run_pipeline(text: str, mode: str, size: int, out_path: str):
    config = load_config()
    tts_service = TTSService(config.triton_url, config.tts)

    session = tts_service.create_session()
    if not session.initialize():
        raise RuntimeError("Failed to initialize TTS session")

    metrics_manager = StreamingMetricsManager(config.streaming)
    pipeline_config = PipelineConfig(
        tts_config=config.tts,
        musetalk_config=config.musetalk,
        streaming_config=config.streaming,
    )
    pipeline = AVPipeline(
        tts_service=tts_service,
        musetalk_service=DummyMuseTalk(),
        metrics_manager=metrics_manager,
        config=pipeline_config,
    )

    text_input_queue = Queue()
    for chunk in build_chunks(text, mode, size):
        text_input_queue.put([chunk])
    text_input_queue.put(None)

    frames = []
    done_event = asyncio.Event()
    error_holder = {"error": None}

    def on_frame(frame):
        frames.append(frame)

    def on_error(msg):
        error_holder["error"] = msg

    def on_complete():
        done_event.set()

    try:
        await pipeline.run(
            text_input_queue=text_input_queue,
            session_id=session.session_id,
            video_enabled=False,
            base_frame_index=0,
            is_generating=lambda: True,
            on_frame=on_frame,
            on_error=on_error,
            on_complete=on_complete,
        )
    finally:
        session.close()

    if error_holder["error"]:
        raise RuntimeError(error_holder["error"])

    if not frames:
        raise RuntimeError("No audio frames captured")

    frames.sort(key=lambda f: f.frame_index)
    audio = np.concatenate([f.audio_samples for f in frames])
    rms = float(np.sqrt(np.mean(audio ** 2)))
    duration = audio.size / float(config.tts.sample_rate)
    print(f"frames={len(frames)} samples={audio.size} duration_s={duration:.3f} rms={rms:.6f}")
    write_wav(out_path, audio, config.tts.sample_rate)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Capture audio from AVPipeline (no websocket)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", default="pipeline.wav", help="Output wav path")
    parser.add_argument("--token-mode", default="chars", choices=["full", "words", "chars"])
    parser.add_argument("--token-size", type=int, default=3, help="Chunk size for token-mode=chars")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.text, args.token_mode, args.token_size, args.out))


if __name__ == "__main__":
    main()
