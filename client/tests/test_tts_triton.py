#!/usr/bin/env python3
import argparse
import queue
import time
import wave
from typing import List, Optional

import numpy as np
import tritonclient.grpc as grpc_client


def split_text_for_streaming(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    words = text.split()
    if not words:
        return ["", ""]
    if len(words) <= 3:
        return [text, "", ""]
    chunks = [" ".join(words[:3])]
    for word in words[3:]:
        chunks.append(" " + word)
    chunks.extend(["", ""])
    return chunks


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


def _wait_for_final(result_queue: queue.Queue, timeout_s: float = 60.0):
    while True:
        msg_type, data = result_queue.get(timeout=timeout_s)
        if msg_type == "error":
            raise RuntimeError(f"Triton error: {data}")
        response = data.get_response()
        params = response.parameters or {}
        if "triton_final_response" in params and params["triton_final_response"].bool_param:
            return
        yield data


def _build_inputs(
    session_id: int,
    text: Optional[str] = None,
    start: bool = False,
    end: bool = False,
    backbone_temp: Optional[float] = None,
    backbone_top_p: Optional[float] = None,
    depth_temp: Optional[float] = None,
    depth_top_p: Optional[float] = None,
) -> List[grpc_client.InferInput]:
    inputs: List[grpc_client.InferInput] = []

    if start:
        inp = grpc_client.InferInput("START", [1], "BOOL")
        inp.set_data_from_numpy(np.array([True], dtype=bool))
        inputs.append(inp)

    if end:
        inp = grpc_client.InferInput("END", [1], "BOOL")
        inp.set_data_from_numpy(np.array([True], dtype=bool))
        inputs.append(inp)

    if text is not None:
        inp = grpc_client.InferInput("TEXTS", [1], "BYTES")
        inp.set_data_from_numpy(np.array([text.encode("utf-8")], dtype=object))
        inputs.append(inp)

    corr = grpc_client.InferInput("CORRID", [1], "INT64")
    corr.set_data_from_numpy(np.array([session_id], dtype=np.int64))
    inputs.append(corr)

    if backbone_temp is not None:
        inp = grpc_client.InferInput("BACKBONE_TEMPERATURE", [1], "FP32")
        inp.set_data_from_numpy(np.array([backbone_temp], dtype=np.float32))
        inputs.append(inp)

    if backbone_top_p is not None:
        inp = grpc_client.InferInput("BACKBONE_TOP_P", [1], "FP32")
        inp.set_data_from_numpy(np.array([backbone_top_p], dtype=np.float32))
        inputs.append(inp)

    if depth_temp is not None:
        inp = grpc_client.InferInput("DEPTH_TEMPERATURE", [1], "FP32")
        inp.set_data_from_numpy(np.array([depth_temp], dtype=np.float32))
        inputs.append(inp)

    if depth_top_p is not None:
        inp = grpc_client.InferInput("DEPTH_TOP_P", [1], "FP32")
        inp.set_data_from_numpy(np.array([depth_top_p], dtype=np.float32))
        inputs.append(inp)

    return inputs


def run_tts_stream(
    url: str,
    text: str,
    sample_rate: int,
    chunk_samples: int,
    backbone_temp: Optional[float],
    backbone_top_p: Optional[float],
    depth_temp: Optional[float],
    depth_top_p: Optional[float],
) -> np.ndarray:
    result_queue: queue.Queue = queue.Queue()

    def callback(result, error):
        if error:
            result_queue.put(("error", error))
        else:
            result_queue.put(("result", result))

    client = grpc_client.InferenceServerClient(url=url)
    client.start_stream(callback=callback)

    session_id = int(time.time() * 1000) & 0x7FFFFFFF
    outputs = [grpc_client.InferRequestedOutput("AUDIO_FRAME")]

    client.async_stream_infer(
        model_name="tts",
        inputs=_build_inputs(session_id, start=True),
        outputs=outputs,
        sequence_id=session_id,
        sequence_start=True,
        sequence_end=False,
        enable_empty_final_response=True,
    )
    for _ in _wait_for_final(result_queue):
        pass

    audio_frames = []
    empty_frames = 0
    frame_count = 0

    for chunk in split_text_for_streaming(text):
        client.async_stream_infer(
            model_name="tts",
            inputs=_build_inputs(
                session_id,
                text=chunk,
                backbone_temp=backbone_temp,
                backbone_top_p=backbone_top_p,
                depth_temp=depth_temp,
                depth_top_p=depth_top_p,
            ),
            outputs=outputs,
            sequence_id=session_id,
            sequence_start=False,
            sequence_end=False,
            enable_empty_final_response=True,
        )

        for result in _wait_for_final(result_queue):
            audio = result.as_numpy("AUDIO_FRAME")
            if audio is None:
                empty_frames += 1
                continue
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                empty_frames += 1
                continue
            frame_count += 1
            if frame_count <= 3 or frame_count % 25 == 0 or audio.size != chunk_samples:
                min_val = float(np.min(audio))
                max_val = float(np.max(audio))
                rms_val = float(np.sqrt(np.mean(audio ** 2)))
                print(
                    f"frame={frame_count} samples={audio.size} expected={chunk_samples} "
                    f"min={min_val:.4f} max={max_val:.4f} rms={rms_val:.4f}"
                )
            audio_frames.append(audio)

    client.async_stream_infer(
        model_name="tts",
        inputs=_build_inputs(session_id, end=True),
        outputs=outputs,
        sequence_id=session_id,
        sequence_start=False,
        sequence_end=True,
        enable_empty_final_response=True,
    )
    for _ in _wait_for_final(result_queue):
        pass

    client.stop_stream()

    audio = np.concatenate(audio_frames) if audio_frames else np.zeros((0,), dtype=np.float32)
    duration_s = audio.size / float(sample_rate) if sample_rate else 0.0
    if audio.size:
        min_val = float(np.min(audio))
        max_val = float(np.max(audio))
        rms_val = float(np.sqrt(np.mean(audio ** 2)))
    else:
        min_val = max_val = rms_val = 0.0

    print(
        f"frames={frame_count} empty_frames={empty_frames} samples={audio.size} "
        f"duration_s={duration_s:.3f} min={min_val:.4f} max={max_val:.4f} rms={rms_val:.4f}"
    )

    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct TTS Triton test")
    parser.add_argument("--url", default="185.151.171.35:54757", help="Triton gRPC URL")
    parser.add_argument("--text", default="გამარჯობა, როგორ შემიძლია დაგეხმაროთ?", help="Text to synthesize")
    parser.add_argument("--out", default="tts_capture.wav", help="Output wav path")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate")
    parser.add_argument("--chunk-samples", type=int, default=1920, help="Expected samples per frame")
    parser.add_argument("--backbone-temp", type=float, default=0.01, help="Backbone temperature")
    parser.add_argument("--backbone-top-p", type=float, default=0.999, help="Backbone top_p")
    parser.add_argument("--depth-temp", type=float, default=0.01, help="Depth temperature")
    parser.add_argument("--depth-top-p", type=float, default=0.999, help="Depth top_p")
    args = parser.parse_args()

    audio = run_tts_stream(
        url=args.url,
        text=args.text,
        sample_rate=args.sample_rate,
        chunk_samples=args.chunk_samples,
        backbone_temp=args.backbone_temp,
        backbone_top_p=args.backbone_top_p,
        depth_temp=args.depth_temp,
        depth_top_p=args.depth_top_p,
    )

    if audio.size == 0:
        raise SystemExit("No audio frames returned from Triton")
    write_wav(args.out, audio, args.sample_rate)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
