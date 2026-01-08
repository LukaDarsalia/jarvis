#!/usr/bin/env python3
import argparse
import os
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from streaming_chunker import split_text_for_streaming
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


def compare_audio(a: np.ndarray, b: np.ndarray) -> dict:
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


def synthesize_tts(tts_service: TTSService, text: str) -> np.ndarray:
    session = tts_service.create_session()
    if not session.initialize():
        raise RuntimeError("Failed to initialize TTS session")

    audio_chunks: list[np.ndarray] = []
    chunks = split_text_for_streaming(text)
    try:
        for chunk in chunks:
            for audio_chunk, _, _ in tts_service.generate_stream([chunk], session_id=session.session_id):
                audio_chunks.append(audio_chunk)
    finally:
        session.close()

    if audio_chunks:
        return np.concatenate(audio_chunks)
    return np.zeros((0,), dtype=np.float32)


def llm_worker(triton: TritonClient, prompt_text: str, first_token: threading.Event) -> None:
    prompt = triton.llm.build_prompt(prompt_text, conversation_history=None)
    for token in triton.llm.generate_stream(prompt):
        if token and not first_token.is_set():
            first_token.set()


def run_overlap_test(
    triton: TritonClient,
    tts_service: TTSService,
    tts_text: str,
    llm_text: str,
    llm_streams: int,
    start_delay_ms: float,
) -> np.ndarray:
    first_token = threading.Event()
    threads: list[threading.Thread] = []

    for _ in range(llm_streams):
        thread = threading.Thread(
            target=llm_worker,
            args=(triton, llm_text, first_token),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    if not first_token.wait(timeout=10.0):
        raise RuntimeError("LLM stream did not produce any tokens within timeout")

    if start_delay_ms > 0:
        time.sleep(start_delay_ms / 1000.0)

    audio = synthesize_tts(tts_service, tts_text)

    for thread in threads:
        thread.join()

    return audio


def print_stats(label: str, audio: np.ndarray, sample_rate: int) -> None:
    duration = audio.size / float(sample_rate) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    print(f"{label}: samples={audio.size} duration_s={duration:.3f} rms={rms:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM overlap impact on TTS audio")
    parser.add_argument(
        "--tts-text",
        default="გამარჯობა! როგორ შემიძლია დაგეხმაროთ? თუ გაქვთ რაიმე შეკითხვა თიბისი ბანკთან დაკავშირებით, ნუ მოგერიდებათ დასმა.",
        help="Text to synthesize with TTS",
    )
    parser.add_argument(
        "--llm-text",
        default="გამარჯობა",
        help="Prompt text to keep LLM stream busy",
    )
    parser.add_argument("--llm-streams", type=int, default=1, help="Number of concurrent LLM streams")
    parser.add_argument("--start-delay-ms", type=float, default=0.0, help="Delay before TTS after LLM starts")
    parser.add_argument("--base-out", default="tts_baseline.wav", help="Baseline wav output")
    parser.add_argument("--overlap-out", default="tts_overlap.wav", help="Overlap wav output")
    parser.add_argument("--no-save", action="store_true", help="Skip writing wav outputs")
    args = parser.parse_args()

    config = load_config()
    triton = TritonClient(
        triton_url=config.triton_url,
        vad_config=config.vad,
        llm_config=config.llm,
        tts_config=config.tts,
        musetalk_config=config.musetalk,
    )
    tts_service = TTSService(config.triton_url, config.tts)

    baseline = synthesize_tts(tts_service, args.tts_text)
    overlap = run_overlap_test(
        triton=triton,
        tts_service=tts_service,
        tts_text=args.tts_text,
        llm_text=args.llm_text,
        llm_streams=max(1, args.llm_streams),
        start_delay_ms=max(0.0, args.start_delay_ms),
    )

    print_stats("baseline", baseline, config.tts.sample_rate)
    print_stats("overlap", overlap, config.tts.sample_rate)

    stats = compare_audio(overlap, baseline)
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
        write_wav(args.base_out, baseline, config.tts.sample_rate)
        write_wav(args.overlap_out, overlap, config.tts.sample_rate)
        print(f"Saved: {args.base_out}")
        print(f"Saved: {args.overlap_out}")


if __name__ == "__main__":
    main()
