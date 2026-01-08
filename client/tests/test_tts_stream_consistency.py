#!/usr/bin/env python3
import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from streaming_chunker import StreamingTTSChunker, split_text_for_streaming
from tts_service import TTSService

os.environ.setdefault("TRITON_URL", "185.151.171.35:54757")


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


def compare_audio(a: np.ndarray, b: np.ndarray):
    if a.size == 0 or b.size == 0:
        return {}
    min_len = min(a.size, b.size)
    a_clip = a[:min_len]
    b_clip = b[:min_len]
    diff_rms = float(np.sqrt(np.mean((a_clip - b_clip) ** 2)))
    denom = float(np.linalg.norm(a_clip) * np.linalg.norm(b_clip))
    corr = float(np.dot(a_clip, b_clip) / denom) if denom > 0 else 0.0
    return {
        "len_a": int(a.size),
        "len_b": int(b.size),
        "len_ratio": float(a.size / b.size) if b.size else 0.0,
        "diff_rms": diff_rms,
        "corr": corr,
    }


def synthesize_streaming(tts_service: TTSService, chunks: list[str], delay_s: float):
    session = tts_service.create_session()
    if not session.initialize():
        raise RuntimeError("Failed to initialize TTS session (streaming)")
    audio_chunks = []
    try:
        for chunk in chunks:
            for audio_chunk, _, _ in tts_service.generate_stream([chunk], session_id=session.session_id):
                audio_chunks.append(audio_chunk)
            if delay_s > 0:
                time.sleep(delay_s)
    finally:
        session.close()
    return np.concatenate(audio_chunks) if audio_chunks else np.zeros((0,), dtype=np.float32)


def synthesize_full(tts_service: TTSService, chunks: list[str]):
    session = tts_service.create_session()
    if not session.initialize():
        raise RuntimeError("Failed to initialize TTS session (full)")
    audio_chunks = []
    try:
        for audio_chunk, _, _ in tts_service.generate_stream(chunks, session_id=session.session_id):
            audio_chunks.append(audio_chunk)
    finally:
        session.close()
    return np.concatenate(audio_chunks) if audio_chunks else np.zeros((0,), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare streaming vs full TTS output")
    parser.add_argument("--text", default="გამარჯობა", help="Text to synthesize")
    parser.add_argument("--token-mode", default="chars", choices=["full", "words", "chars"])
    parser.add_argument("--token-size", type=int, default=3, help="Chunk size for token-mode=chars")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="Delay between streamed chunks")
    parser.add_argument("--stream-out", default="tts_stream.wav", help="Streaming output wav")
    parser.add_argument("--full-out", default="tts_full.wav", help="Full output wav")
    parser.add_argument("--no-save", action="store_true", help="Skip writing wav files")
    args = parser.parse_args()

    config = load_config()
    tts_service = TTSService(config.triton_url, config.tts)

    chunks = build_chunks(args.text, args.token_mode, args.token_size)
    print(f"chunks={len(chunks)} token_mode={args.token_mode} delay_ms={args.delay_ms}")

    delay_s = max(args.delay_ms, 0.0) / 1000.0
    audio_stream = synthesize_streaming(tts_service, chunks, delay_s)
    audio_full = synthesize_full(tts_service, chunks)

    rms_stream = float(np.sqrt(np.mean(audio_stream ** 2))) if audio_stream.size else 0.0
    rms_full = float(np.sqrt(np.mean(audio_full ** 2))) if audio_full.size else 0.0
    dur_stream = audio_stream.size / float(config.tts.sample_rate) if audio_stream.size else 0.0
    dur_full = audio_full.size / float(config.tts.sample_rate) if audio_full.size else 0.0
    print(
        f"stream: samples={audio_stream.size} duration_s={dur_stream:.3f} rms={rms_stream:.6f}"
    )
    print(
        f"full: samples={audio_full.size} duration_s={dur_full:.3f} rms={rms_full:.6f}"
    )

    stats = compare_audio(audio_stream, audio_full)
    if stats:
        print(
            "compare: "
            f"len_a={stats['len_a']} "
            f"len_b={stats['len_b']} "
            f"len_ratio={stats['len_ratio']:.3f} "
            f"diff_rms={stats['diff_rms']:.6f} "
            f"corr={stats['corr']:.6f}"
        )

    if not args.no_save:
        write_wav(args.stream_out, audio_stream, config.tts.sample_rate)
        write_wav(args.full_out, audio_full, config.tts.sample_rate)
        print(f"Saved: {args.stream_out}")
        print(f"Saved: {args.full_out}")


if __name__ == "__main__":
    main()
