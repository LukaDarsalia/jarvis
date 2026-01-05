#!/usr/bin/env python3
import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from streaming_chunker import StreamingTTSChunker
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


def main():
    parser = argparse.ArgumentParser(description="Capture audio via TTSService streaming")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", default="tts_service.wav", help="Output wav path")
    parser.add_argument("--token-mode", default="chars", choices=["full", "words", "chars"])
    parser.add_argument("--token-size", type=int, default=3, help="Chunk size for token-mode=chars")
    args = parser.parse_args()

    config = load_config()
    tts_service = TTSService(config.triton_url, config.tts)

    session = tts_service.create_session()
    if not session.initialize():
        raise SystemExit("Failed to initialize TTS session")

    chunks = build_chunks(args.text, args.token_mode, args.token_size)
    audio_chunks = []
    frame_count = 0

    try:
        for chunk in chunks:
            for audio, word, metrics in tts_service.generate_stream([chunk], session_id=session.session_id):
                audio_chunks.append(audio)
                frame_count += 1
    finally:
        session.close()

    if not audio_chunks:
        raise SystemExit("No audio frames captured")

    audio = np.concatenate(audio_chunks)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"frames={frame_count} samples={audio.size} rms={rms:.6f}")
    write_wav(args.out, audio, config.tts.sample_rate)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
