#!/usr/bin/env python3
import argparse
import os
import sys
import wave
from statistics import median

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from streaming_chunker import StreamingTTSChunker, split_text_for_streaming
from tts_service import TTSService
from triton_services import TritonClient
os.environ.setdefault("TRITON_URL", "185.151.171.35:54757")

def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())

def analyze_frame_diffs(frames: list[np.ndarray]):
    if len(frames) < 2:
        return {}
    diffs = []
    exact_dupes = 0
    near_dupes = 0
    for prev, cur in zip(frames[:-1], frames[1:]):
        if prev.shape != cur.shape:
            continue
        if np.array_equal(prev, cur):
            exact_dupes += 1
            diffs.append(0.0)
            continue
        mean_abs = float(np.mean(np.abs(cur - prev)))
        diffs.append(mean_abs)
        if mean_abs < 1e-5:
            near_dupes += 1
    if not diffs:
        return {}
    return {
        "pairs": len(diffs),
        "min_diff": min(diffs),
        "median_diff": median(diffs),
        "max_diff": max(diffs),
        "exact_dupes": exact_dupes,
        "near_dupes": near_dupes,
    }

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream LLM tokens into TTS and save audio")
    parser.add_argument("--text", default="გამარჯობა", help="User prompt for LLM")
    parser.add_argument("--out", default="llm_tts.wav", help="Output wav path")
    parser.add_argument("--full-out", default="llm_tts_full.wav", help="Output wav path for full-text chunks")
    parser.add_argument("--compare-full", action="store_true", help="Also synthesize using full-text chunks")
    parser.add_argument("--tokens-out", default="llm_tokens.txt", help="Path to save LLM tokens")
    parser.add_argument("--chunks-out", default="tts_chunks.txt", help="Path to save TTS chunks")
    args = parser.parse_args()

    config = load_config()

    triton = TritonClient(
        triton_url=config.triton_url,
        vad_config=config.vad,
        llm_config=config.llm,
        tts_config=config.tts,
        musetalk_config=config.musetalk,
    )

    prompt = triton.llm.build_prompt(args.text, conversation_history=None)

    tts_service = TTSService(config.triton_url, config.tts)
    session = tts_service.create_session()
    if not session.initialize():
        raise SystemExit("Failed to initialize TTS session")

    chunker = StreamingTTSChunker()
    audio_chunks = []
    tokens = []
    chunks = []

    try:
        for token in triton.llm.generate_stream(prompt):
            tokens.append(token)
            for chunk in chunker.push_token(token):
                chunks.append(chunk)
                for audio, _, _ in tts_service.generate_stream([chunk], session_id=session.session_id):
                    audio_chunks.append(audio)

        for chunk in chunker.finalize():
            chunks.append(chunk)
            for audio, _, _ in tts_service.generate_stream([chunk], session_id=session.session_id):
                audio_chunks.append(audio)
    finally:
        session.close()

    if not audio_chunks:
        raise SystemExit("No audio frames captured")

    audio = np.concatenate(audio_chunks)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    duration = audio.size / float(config.tts.sample_rate)
    print(f"tokens={len(tokens)} chunks={len(chunks)} frames={len(audio_chunks)}")
    print(f"samples={audio.size} duration_s={duration:.3f} rms={rms:.6f}")
    frame_diff_stats = analyze_frame_diffs(audio_chunks)
    if frame_diff_stats:
        print(
            "frame_diffs: "
            f"pairs={frame_diff_stats['pairs']} "
            f"min={frame_diff_stats['min_diff']:.6f} "
            f"median={frame_diff_stats['median_diff']:.6f} "
            f"max={frame_diff_stats['max_diff']:.6f} "
            f"exact_dupes={frame_diff_stats['exact_dupes']} "
            f"near_dupes={frame_diff_stats['near_dupes']}"
        )

    write_wav(args.out, audio, config.tts.sample_rate)
    print(f"Saved: {args.out}")

    with open(args.tokens_out, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(repr(token))
            f.write("\n")
    print(f"Saved: {args.tokens_out}")

    with open(args.chunks_out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(repr(chunk))
            f.write("\n")
    print(f"Saved: {args.chunks_out}")

    full_text = "".join(tokens).strip()
    expected_chunks = split_text_for_streaming(full_text)
    if expected_chunks != chunks:
        print("Chunk mismatch vs full-text split:")
        print(f"full_text: {full_text}")
        print(f"expected_chunks={expected_chunks}")
    else:
        print("Chunk list matches full-text split.")

    if args.compare_full:
        session_full = tts_service.create_session()
        if not session_full.initialize():
            raise SystemExit("Failed to initialize TTS session for full-text comparison")
        audio_full_chunks = []
        try:
            for chunk in expected_chunks:
                for audio, _, _ in tts_service.generate_stream([chunk], session_id=session_full.session_id):
                    audio_full_chunks.append(audio)
        finally:
            session_full.close()

        if audio_full_chunks:
            audio_full = np.concatenate(audio_full_chunks)
            rms_full = float(np.sqrt(np.mean(audio_full ** 2)))
            duration_full = audio_full.size / float(config.tts.sample_rate)
            print(f"full_frames={len(audio_full_chunks)} samples={audio_full.size} "
                  f"duration_s={duration_full:.3f} rms={rms_full:.6f}")
            compare_stats = compare_audio(audio, audio_full)
            if compare_stats:
                print(
                    "compare_full: "
                    f"len_a={compare_stats['len_a']} "
                    f"len_b={compare_stats['len_b']} "
                    f"len_ratio={compare_stats['len_ratio']:.3f} "
                    f"diff_rms={compare_stats['diff_rms']:.6f} "
                    f"corr={compare_stats['corr']:.6f}"
                )
            write_wav(args.full_out, audio_full, config.tts.sample_rate)
            print(f"Saved: {args.full_out}")


if __name__ == "__main__":
    main()
