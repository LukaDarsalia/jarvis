import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from tts_generator import TTSGenerator


@dataclass
class TTSConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    decoder_temperature: float = 0.8
    decoder_top_p: float = 0.9
    model_id: str = "/local_models/tts_model/csm-1b-base"
    model_path: str = "/local_models/tts_model/georgian-csm-1b"
    device: str = "cuda"
    max_steps: int = 125 // 2


class TritonPythonModel:
    def initialize(self, args):
        try:
            pb_utils.Logger.log_info("=== INITIALIZING STREAMING TTS (DECOUPLED) ===")
            self.model_config = json.loads(args['model_config'])

            params = self.model_config.get('parameters', {})

            self.output_sample_rate = int(self._get_param(params, 'output_sample_rate', '24000'))
            self.default_target_sample_rate = int(self._get_param(params, 'default_sample_rate', '24000'))

            # ~1920 samples @ 24k, expressed as seconds
            self.stream_chunk_duration = 1920.0 / float(self.output_sample_rate)

            self.config = TTSConfig()
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

            pb_utils.Logger.log_info(f"Initializing TTSGenerator on device: {self.config.device}")
            self.tts_generator = TTSGenerator(
                model_id=self.config.model_id,
                model_path=self.config.model_path,
                device=self.config.device,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                decoder_temperature=self.config.decoder_temperature,
                decoder_top_p=self.config.decoder_top_p,
                compile_model=True,
                reference_audio_path="/local_models/tts_model/georgian-csm-1b/context_audio_for_inference.wav",
                reference_json_path="/local_models/tts_model/georgian-csm-1b/context_text_for_inference.json",
            )

            warmup_enabled = os.environ.get("TTS_WARMUP", "1") != "0"
            if warmup_enabled:
                warmup_text = os.environ.get("TTS_WARMUP_TEXT", "").strip() or None
                warmup_steps = int(os.environ.get("TTS_WARMUP_STEPS", "12"))
                warmup_rounds = int(os.environ.get("TTS_WARMUP_ROUNDS", "2"))
                warmup_start = time.perf_counter()
                try:
                    ok = self.tts_generator.warmup(
                        speaker_id=0,
                        text=warmup_text,
                        max_steps=warmup_steps,
                        max_rounds=warmup_rounds,
                    )
                    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
                    pb_utils.Logger.log_info(
                        f"TTS warmup {'completed' if ok else 'skipped'} "
                        f"in {warmup_ms:.1f}ms (steps={warmup_steps}, rounds={warmup_rounds})"
                    )
                except Exception as exc:
                    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
                    pb_utils.Logger.log_warn(
                        f"TTS warmup failed after {warmup_ms:.1f}ms: {exc}"
                    )

            pb_utils.Logger.log_info("TTS Triton model initialized")
            self.log_every_n_steps = int(os.environ.get("TTS_LOG_EVERY_N_STEPS", "10"))

        except Exception as e:
            msg = f"Failed to initialize: {e}\n{traceback.format_exc()}"
            pb_utils.Logger.log_error(msg)
            raise

    def _get_param(self, params, key, default):
        if key in params:
            value = params[key]['string_value']
        else:
            value = default
        pb_utils.Logger.log_info(f"Parameter {key}: {value}")
        return value

    def _get_bool_input(self, request, name: str, default: bool = False) -> bool:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return bool(arr.squeeze())

    def _get_int_input(self, request, name: str, default: Optional[int] = None) -> Optional[int]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return int(arr.squeeze())

    def _get_text_list(self, request) -> List[str]:
        tensor = pb_utils.get_input_tensor_by_name(request, "TEXTS")
        if tensor is None:
            return []
        arr = tensor.as_numpy()  # shape (N,) or (N, 1)
        if arr.size == 0:
            return []
        arr = arr.reshape(-1)
        return [x.decode("utf-8") for x in arr]

    def _get_float_input(self, request, name: str, default: Optional[float] = None) -> Optional[float]:
        """
        Safely extracts a single float (FP32) value from an input tensor.
        
        Args:
            request: The inference request object.
            name: The name of the input tensor (e.g., "BACKBONE_TEMPERATURE").
            default: The default value to return if the tensor is not provided or is empty.

        Returns:
            The float value, or the default value if the tensor is missing or empty.
        """
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return float(arr.squeeze())

    def _send_final(self, sender):
        sender.send(self._empty_response(), flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def execute(self, requests):
        """
        Decoupled mode:
        - For each request, get a response sender
        - Stream AUDIO_FRAME chunks via sender.send(...)
        - Finish with sender.send_final(...)
        """
        for request in requests:
            sender = request.get_response_sender()
            self._handle_request_decoupled(request, sender)

    def _handle_request_decoupled(self, request, sender):
        # inputs
        texts = self._get_text_list(request)
        is_start = self._get_bool_input(request, "START", False)
        is_end = self._get_bool_input(request, "END", False)
        seq_id = self._get_int_input(request, "CORRID", None)
        temperature = self._get_float_input(request, "BACKBONE_TEMPERATURE", None)
        topp = self._get_float_input(request, "BACKBONE_TOP_P", None)
        depth_temperature = self._get_float_input(request, "DEPTH_TEMPERATURE", None)
        depth_topp = self._get_float_input(request, "DEPTH_TOP_P", None)

        # Session ID is must!
        if seq_id is None:
            self._send_final(sender)
            raise RuntimeError("CORRID input is required")

        # Session has to be initialized, so using any flag other than init_cache is wrong!
        sessions = self.tts_generator.get_active_sessions()
        if not is_start and seq_id not in sessions:
            self._send_final(sender)
            raise RuntimeError("You must firstly initialize cache before starting!")

        # Handles init cache. If session ids are more than supposed number, raises error!
        if is_start:
            t0 = time.perf_counter()
            is_initialized = self.tts_generator.initialize_session(seq_id)
            init_ms = (time.perf_counter() - t0) * 1000.0
            self._send_final(sender)
            if not is_initialized:
                raise RuntimeError(f"[Seq {seq_id}] Failed to start session: max concurrent reached")
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Cache initialized successfully in {init_ms:.1f}ms")
            return

        # Handles ending. Ends session
        if is_end:
            t0 = time.perf_counter()
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Session successfully ended!")
            self.tts_generator.end_session(seq_id)
            end_ms = (time.perf_counter() - t0) * 1000.0
            self._send_final(sender)
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Session cleanup completed in {end_ms:.1f}ms")
            return

        pb_utils.Logger.log_info(f"Appends intermediate texts: {texts}")
        append_start = time.perf_counter()
        ok = self.tts_generator.append_texts(seq_id, texts, speaker_id=0)
        append_ms = (time.perf_counter() - append_start) * 1000.0
        if not ok:
            self._send_final(sender)
            raise RuntimeError(f"[Seq {seq_id}] session is not active!")
        pb_utils.Logger.log_info(f"[Seq {seq_id}] append_texts_ms={append_ms:.1f}")

        req_start = time.perf_counter()
        total_steps = 0
        last_audio_ms = 0.0
        interval_steps = 0

        # Processes one word audio
        # If model loops and never returns eos token, max_steps will stop it!
        for cur_step in range(self.config.max_steps):
            step_start = time.perf_counter()
            is_complete = self.tts_generator.step_session(
                seq_id,
                max_steps=1,
                temperature=temperature,
                topp=topp,
                depth_temperature=depth_temperature,
                depth_topp=depth_topp
            )
            step_ms = (time.perf_counter() - step_start) * 1000.0
            total_steps += 1
            interval_steps += 1

            # Get all codes so far
            codes = self.tts_generator.get_session_audio(seq_id, return_codes=True)
            if codes is None:
                continue

            # Decode full audio from all codes
            decode_start = time.perf_counter()
            audio_24k = self.tts_generator._decode_audio(codes)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            self.tts_generator.record_codec_time(seq_id, decode_ms)
            audio_24k = audio_24k.cpu().float()

            audio_np = audio_24k.numpy()
            last_audio_ms = (len(audio_np) / float(self.output_sample_rate)) * 1000.0

            # Chunk length = 1920 @ 24k scaled by target_sr
            chunk_len = int(self.stream_chunk_duration * float(self.output_sample_rate))

            # stream last chunk_len samples
            chunk_np = audio_np[-chunk_len:].astype(np.float32)
            audio_tensor = pb_utils.Tensor("AUDIO_FRAME", chunk_np)
            sender.send(pb_utils.InferenceResponse(
                output_tensors=[audio_tensor]
            ))

            if self.log_every_n_steps > 0 and (interval_steps >= self.log_every_n_steps or is_complete):
                perf = self.tts_generator.get_session_perf(seq_id, reset=True)
                gen_ms = (time.perf_counter() - req_start) * 1000.0
                rtf = (gen_ms / last_audio_ms) if last_audio_ms > 0 else 0.0
                if perf is not None:
                    bb_inject_avg = perf.bb_inject_ms / perf.inject_calls if perf.inject_calls else 0.0
                    bb_audio_avg = perf.bb_audio_ms / perf.audio_calls if perf.audio_calls else 0.0
                    dd_avg = perf.dd_ms / perf.dd_calls if perf.dd_calls else 0.0
                    lm_avg = perf.lm_ms / perf.lm_calls if perf.lm_calls else 0.0
                    codec_avg = perf.codec_ms / perf.codec_calls if perf.codec_calls else 0.0
                    pb_utils.Logger.log_info(
                        f"[Seq {seq_id}] steps={total_steps} step_ms={step_ms:.1f} decode_ms={decode_ms:.1f} "
                        f"audio_ms={last_audio_ms:.1f} rtf={rtf:.3f} "
                        f"bb_inject_ms(avg)={bb_inject_avg:.2f} bb_audio_ms(avg)={bb_audio_avg:.2f} "
                        f"dd_ms(avg)={dd_avg:.2f} lm_ms(avg)={lm_avg:.2f} codec_ms(avg)={codec_avg:.2f}"
                    )
                interval_steps = 0

            # Means model returned eos token
            if is_complete:
                break

            # Means max_steps stopped the process. Because of that, we must increase word pointer
            if cur_step == self.config.max_steps - 1:
                self.tts_generator.increment_state_counter(seq_id)

        # Ends streaming
        self._send_final(sender)


    def _empty_response(self):
        audio = np.zeros((0,), dtype=np.float32)
        audio_tensor = pb_utils.Tensor("AUDIO_FRAME", audio)
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor])

    def finalize(self):
        try:
            pb_utils.Logger.log_info("Finalizing TTS model...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            msg = f"Error in finalize: {e}\n{traceback.format_exc()}"
            pb_utils.Logger.log_error(msg)
