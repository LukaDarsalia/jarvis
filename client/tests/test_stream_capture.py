import argparse
import asyncio
import base64
import io
import time
from fractions import Fraction
from statistics import median

import numpy as np
import websockets

try:
    import av
    from PIL import Image
except Exception as exc:  # pragma: no cover - runtime dependency check
    av = None
    Image = None


async def capture_stream(ws_url: str, text: str, timeout_s: float = 60.0):
    audio_frames = {}
    video_frames = {}
    audio_meta = {}
    duplicate_indices = 0
    arrival_order = []
    got_tts_complete = False
    got_video_complete = False
    connected = False
    llm_text = ""

    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        import json
        start = time.time()
        await ws.send(json.dumps({"type": "text_input", "text": text}))

        while time.time() - start < timeout_s:
            msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            data = None
            try:
                data = json.loads(msg)
            except Exception:
                continue

            msg_type = data.get("type")
            if msg_type == "connected":
                connected = True
            elif msg_type == "synced_av_frame":
                idx = int(data.get("frame_index", 0))
                arrival_order.append(idx)
                if idx in audio_frames or idx in video_frames:
                    duplicate_indices += 1
                if data.get("audio"):
                    audio_frames[idx] = base64.b64decode(data["audio"])
                    audio_meta[idx] = {
                        "audio_samples": int(data.get("audio_samples", 0)),
                        "audio_crc32": int(data.get("audio_crc32", 0)),
                    }
                if data.get("frame"):
                    video_frames[idx] = base64.b64decode(data["frame"])
            elif msg_type == "tts_complete":
                got_tts_complete = True
            elif msg_type == "video_complete":
                got_video_complete = True
            elif msg_type == "llm_complete":
                llm_text = data.get("text", "") or llm_text

            if got_tts_complete and got_video_complete:
                break

    return {
        "connected": connected,
        "audio_frames": audio_frames,
        "video_frames": video_frames,
        "audio_meta": audio_meta,
        "duplicate_indices": duplicate_indices,
        "arrival_order": arrival_order,
        "tts_complete": got_tts_complete,
        "video_complete": got_video_complete,
        "llm_text": llm_text,
    }


def build_stream_arrays(audio_frames: dict, video_frames: dict, fps: int, sample_rate: int):
    samples_per_frame = int(sample_rate / fps)
    if not audio_frames and not video_frames:
        raise RuntimeError("No frames captured")

    max_index = max(
        [0]
        + list(audio_frames.keys())
        + list(video_frames.keys())
    )

    audio_pcm = []
    missing_audio = 0
    short_audio = 0
    long_audio = 0
    for idx in range(max_index + 1):
        if idx in audio_frames:
            pcm = np.frombuffer(audio_frames[idx], dtype=np.float32)
        else:
            pcm = np.zeros(samples_per_frame, dtype=np.float32)
            missing_audio += 1

        if pcm.size < samples_per_frame:
            short_audio += 1
            pad = np.zeros(samples_per_frame - pcm.size, dtype=np.float32)
            pcm = np.concatenate([pcm, pad])
        elif pcm.size > samples_per_frame:
            long_audio += 1
            pcm = pcm[:samples_per_frame]

        audio_pcm.append(pcm)

    audio_pcm = np.concatenate(audio_pcm) if audio_pcm else np.zeros((0,), dtype=np.float32)

    video_seq = []
    last_frame = None
    missing_video = 0
    for idx in range(max_index + 1):
        if idx in video_frames:
            last_frame = video_frames[idx]
        if last_frame is None:
            missing_video += 1
            continue
        video_seq.append((idx, last_frame))

    stats = {
        "max_index": max_index,
        "missing_audio": missing_audio,
        "short_audio": short_audio,
        "long_audio": long_audio,
        "missing_video": missing_video,
    }

    return audio_pcm, video_seq, stats


def write_mp4(output_path: str, audio_frames: dict, video_frames: dict, fps: int = 25, sample_rate: int = 24000):
    if av is None or Image is None:
        raise RuntimeError("PyAV/Pillow not available; cannot write mp4")

    audio_pcm, video_seq, stats = build_stream_arrays(audio_frames, video_frames, fps, sample_rate)

    output = av.open(output_path, mode="w")

    vstream = None
    if video_seq:
        _, first_bytes = video_seq[0]
        img = Image.open(io.BytesIO(first_bytes)).convert("RGB")
        vstream = output.add_stream("libx264", rate=fps)
        vstream.width = img.width
        vstream.height = img.height
        vstream.pix_fmt = "yuv420p"

    astream = None
    if audio_pcm.size > 0:
        astream = output.add_stream("aac", rate=sample_rate)
        astream.layout = "mono"

    audio_pts = 0
    audio_time_base = Fraction(1, sample_rate)
    video_time_base = Fraction(1, fps)

    for idx, frame_bytes in video_seq:
        if not vstream:
            break
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        frame = av.VideoFrame.from_image(img)
        frame.pts = idx
        frame.time_base = video_time_base
        for packet in vstream.encode(frame):
            output.mux(packet)

    if astream and audio_pcm.size > 0:
        frame_size = astream.codec_context.frame_size or 1024
        pcm = np.clip(audio_pcm, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        total_samples = pcm_i16.size
        for start in range(0, total_samples, frame_size):
            chunk = pcm_i16[start:start + frame_size]
            if chunk.size < frame_size:
                pad = np.zeros(frame_size - chunk.size, dtype=np.int16)
                chunk = np.concatenate([chunk, pad])
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = audio_pts
            frame.time_base = audio_time_base
            audio_pts += frame_size
            for packet in astream.encode(frame):
                output.mux(packet)

    if vstream:
        for packet in vstream.encode():
            output.mux(packet)
    if astream:
        for packet in astream.encode():
            output.mux(packet)

    output.close()
    return stats


def write_wav(output_path: str, audio_frames: dict, video_frames: dict, fps: int = 25, sample_rate: int = 24000):
    audio_pcm, _, stats = build_stream_arrays(audio_frames, video_frames, fps, sample_rate)
    pcm = np.clip(audio_pcm, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    import wave
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return stats


def analyze_audio_frames(audio_frames: dict, fps: int = 25, sample_rate: int = 24000):
    samples_per_frame = int(sample_rate / fps)
    if not audio_frames:
        return {}

    max_index = max(audio_frames.keys())
    rms_values = []
    duplicate_frames = 0
    near_duplicate_frames = 0
    silent_frames = 0
    long_frames = 0
    prev_pcm = None

    for idx in range(max_index + 1):
        if idx in audio_frames:
            pcm = np.frombuffer(audio_frames[idx], dtype=np.float32)
        else:
            pcm = np.zeros(samples_per_frame, dtype=np.float32)

        if pcm.size < samples_per_frame:
            pad = np.zeros(samples_per_frame - pcm.size, dtype=np.float32)
            pcm = np.concatenate([pcm, pad])
        elif pcm.size > samples_per_frame:
            long_frames += 1
            pcm = pcm[:samples_per_frame]

        rms = float(np.sqrt(np.mean(pcm ** 2))) if pcm.size else 0.0
        rms_values.append(rms)
        if rms < 1e-4:
            silent_frames += 1

        if prev_pcm is not None and pcm.size == prev_pcm.size:
            if np.array_equal(pcm, prev_pcm):
                duplicate_frames += 1
            else:
                mean_abs_diff = float(np.mean(np.abs(pcm - prev_pcm)))
                if mean_abs_diff < 1e-5:
                    near_duplicate_frames += 1
        prev_pcm = pcm

    return {
        "frames": max_index + 1,
        "min_rms": min(rms_values) if rms_values else 0.0,
        "median_rms": median(rms_values) if rms_values else 0.0,
        "max_rms": max(rms_values) if rms_values else 0.0,
        "silent_frames": silent_frames,
        "duplicate_frames": duplicate_frames,
        "near_duplicate_frames": near_duplicate_frames,
        "long_frames": long_frames,
    }


def verify_audio_integrity(audio_frames: dict, audio_meta: dict, fps: int = 25, sample_rate: int = 24000):
    import zlib
    samples_per_frame = int(sample_rate / fps)
    crc_mismatch = 0
    size_mismatch = 0
    missing_meta = 0
    for idx, raw in audio_frames.items():
        meta = audio_meta.get(idx)
        if not meta:
            missing_meta += 1
            continue
        expected_samples = int(meta.get("audio_samples", 0))
        expected_crc = int(meta.get("audio_crc32", 0))
        if expected_samples and expected_samples != len(raw) // 4:
            size_mismatch += 1
        if expected_crc:
            actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                crc_mismatch += 1
    return {
        "crc_mismatch": crc_mismatch,
        "size_mismatch": size_mismatch,
        "missing_meta": missing_meta,
        "samples_per_frame": samples_per_frame,
    }


def analyze_arrival_order(arrival_order: list):
    if not arrival_order:
        return {}
    out_of_order = 0
    prev = arrival_order[0]
    for idx in arrival_order[1:]:
        if idx < prev:
            out_of_order += 1
        prev = idx
    return {"out_of_order": out_of_order, "received": len(arrival_order)}


def main():
    parser = argparse.ArgumentParser(description="Capture AV stream and save as mp4")
    parser.add_argument("--ws", default="ws://localhost:8080/ws", help="WebSocket URL")
    parser.add_argument("--text", default="გამარჯობა", help="Text to send")
    parser.add_argument("--out", default="capture.mp4", help="Output mp4 path")
    parser.add_argument("--out-wav", default="", help="Optional wav path for audio-only output")
    parser.add_argument("--timeout", type=float, default=60.0, help="Capture timeout seconds")
    args = parser.parse_args()

    result = asyncio.run(capture_stream(args.ws, args.text, args.timeout))
    if not result["connected"]:
        raise SystemExit("Did not receive connected message")

    print(
        f"Captured audio frames: {len(result['audio_frames'])}, "
        f"video frames: {len(result['video_frames'])}, "
        f"tts_complete: {result['tts_complete']}, video_complete: {result['video_complete']}"
    )
    if result.get("duplicate_indices"):
        print(f"Duplicate indices received: {result['duplicate_indices']}")

    stats = write_mp4(args.out, result["audio_frames"], result["video_frames"])
    print(f"Saved: {args.out}")
    print(
        "Stats: "
        f"max_index={stats['max_index']} "
        f"missing_audio={stats['missing_audio']} "
        f"short_audio={stats['short_audio']} "
        f"long_audio={stats['long_audio']} "
        f"missing_video={stats['missing_video']}"
    )
    if args.out_wav:
        wav_stats = write_wav(args.out_wav, result["audio_frames"], result["video_frames"])
        print(f"Saved: {args.out_wav}")
        print(
            "WAV stats: "
            f"max_index={wav_stats['max_index']} "
            f"missing_audio={wav_stats['missing_audio']} "
            f"short_audio={wav_stats['short_audio']} "
            f"long_audio={wav_stats['long_audio']} "
            f"missing_video={wav_stats['missing_video']}"
        )
    audio_diag = analyze_audio_frames(result["audio_frames"])
    if audio_diag:
        print(
            "Audio diagnostics: "
            f"frames={audio_diag['frames']} "
            f"min_rms={audio_diag['min_rms']:.6f} "
            f"median_rms={audio_diag['median_rms']:.6f} "
            f"max_rms={audio_diag['max_rms']:.6f} "
            f"silent_frames={audio_diag['silent_frames']} "
            f"duplicate_frames={audio_diag['duplicate_frames']} "
            f"near_duplicate_frames={audio_diag['near_duplicate_frames']} "
            f"long_frames={audio_diag['long_frames']}"
        )
    integrity = verify_audio_integrity(result["audio_frames"], result["audio_meta"])
    if integrity:
        print(
            "Integrity: "
            f"crc_mismatch={integrity['crc_mismatch']} "
            f"size_mismatch={integrity['size_mismatch']} "
            f"missing_meta={integrity['missing_meta']}"
        )
    arrival_diag = analyze_arrival_order(result.get("arrival_order", []))
    if arrival_diag:
        print(
            "Arrival: "
            f"received={arrival_diag['received']} "
            f"out_of_order={arrival_diag['out_of_order']}"
        )
    if result.get("llm_text"):
        print(f"LLM response text: {result['llm_text']}")


if __name__ == "__main__":
    main()
