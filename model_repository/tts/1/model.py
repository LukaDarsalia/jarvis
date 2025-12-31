import json
import os
import threading
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

            self._session_state_lock = threading.Lock()
            self._session_prev_frames: Dict[int, int] = {}
            self._session_audio_buffers: Dict[int, np.ndarray] = {}
            self._session_chunk_counters: Dict[int, int] = {}
            self._log_every_n_chunks = 50

            pb_utils.Logger.log_info("TTS Triton model initialized")

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
            is_initialized = self.tts_generator.initialize_session(seq_id)
            self._send_final(sender)
            if not is_initialized:
                raise RuntimeError(f"[Seq {seq_id}] Failed to start session: max concurrent reached")
            with self._session_state_lock:
                self._session_prev_frames[seq_id] = 0
                self._session_audio_buffers[seq_id] = np.zeros((0,), dtype=np.float32)
                self._session_chunk_counters[seq_id] = 0
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Cache initialized successfully")
            return

        # Handles ending. Ends session
        if is_end:
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Session successfully ended!")
            self.tts_generator.end_session(seq_id)
            with self._session_state_lock:
                self._session_prev_frames.pop(seq_id, None)
                self._session_audio_buffers.pop(seq_id, None)
                self._session_chunk_counters.pop(seq_id, None)
            self._send_final(sender)
            return

        pb_utils.Logger.log_info(f"Appends intermediate texts: {texts}")
        ok = self.tts_generator.append_texts(seq_id, texts, speaker_id=0)
        if not ok:
            self._send_final(sender)
            raise RuntimeError(f"[Seq {seq_id}] session is not active!")

        # Processes one word audio
        # If model loops and never returns eos token, max_steps will stop it!
        with self._session_state_lock:
            prev_num_frames = self._session_prev_frames.get(seq_id, 0)
            audio_buf = self._session_audio_buffers.get(seq_id, np.zeros((0,), dtype=np.float32))

        # Chunk length = 1920 @ 24k scaled by target_sr
        chunk_len = int(self.stream_chunk_duration * float(self.output_sample_rate))

        for cur_step in range(self.config.max_steps):
            is_complete = self.tts_generator.step_session(
                seq_id,
                max_steps=1,
                temperature=temperature,
                topp=topp,
                depth_temperature=depth_temperature,
                depth_topp=depth_topp
            )

            # Get all codes so far
            codes = self.tts_generator.get_session_audio(seq_id, return_codes=True)
            if codes is None:
                continue
            cur_num_frames = codes.shape[0]
            if cur_num_frames <= prev_num_frames:
                if is_complete:
                    break
                continue
            if cur_num_frames > prev_num_frames + 1:
                pb_utils.Logger.log_info(
                    f"[Seq {seq_id}] Multiple frames generated at once: prev={prev_num_frames} cur={cur_num_frames}"
                )

            # Decode full audio and stream only the newly generated frames.
            full_audio = self.tts_generator._decode_audio(codes)
            if full_audio is not None:
                audio_np = full_audio.cpu().float().numpy().astype(np.float32)
                delta_frames = cur_num_frames - prev_num_frames
                if delta_frames > 0:
                    tail_samples = delta_frames * chunk_len
                    if tail_samples > 0:
                        if tail_samples > audio_np.size:
                            tail_samples = audio_np.size
                        delta_np = audio_np[-tail_samples:]
                        if delta_np.size > 0:
                            max_abs = float(np.max(np.abs(delta_np)))
                            if max_abs < 1e-4:
                                pb_utils.Logger.log_info(
                                    f"[Seq {seq_id}] Audio chunk near-silent: samples={delta_np.size} max_abs={max_abs:.6f}"
                                )
                            elif max_abs > 1.2:
                                pb_utils.Logger.log_info(
                                    f"[Seq {seq_id}] Audio chunk high amplitude: samples={delta_np.size} max_abs={max_abs:.3f}"
                                )
                        if audio_buf.size == 0:
                            audio_buf = delta_np
                        else:
                            audio_buf = np.concatenate([audio_buf, delta_np])

            force_flush = bool(is_complete)
            while audio_buf.size >= chunk_len or (force_flush and audio_buf.size > 0):
                if audio_buf.size >= chunk_len:
                    chunk_np = audio_buf[:chunk_len]
                    audio_buf = audio_buf[chunk_len:]
                else:
                    chunk_np = audio_buf
                    audio_buf = np.zeros((0,), dtype=np.float32)
                with self._session_state_lock:
                    chunk_counter = self._session_chunk_counters.get(seq_id, 0) + 1
                    self._session_chunk_counters[seq_id] = chunk_counter
                log_chunk = (
                    chunk_counter <= 3
                    or chunk_counter % self._log_every_n_chunks == 0
                    or chunk_np.size != chunk_len
                )
                if log_chunk and chunk_np.size > 0:
                    min_val = float(np.min(chunk_np))
                    max_val = float(np.max(chunk_np))
                    rms_val = float(np.sqrt(np.mean(chunk_np ** 2))) if chunk_np.size > 0 else 0.0
                    nan_count = int(np.isnan(chunk_np).sum())
                    pb_utils.Logger.log_info(
                        f"[Seq {seq_id}] AUDIO_FRAME {chunk_counter}: samples={chunk_np.size} "
                        f"expected={chunk_len} min={min_val:.4f} max={max_val:.4f} "
                        f"rms={rms_val:.4f} nan={nan_count} prev_frames={prev_num_frames} "
                        f"cur_frames={cur_num_frames}"
                    )
                audio_tensor = pb_utils.Tensor("AUDIO_FRAME", chunk_np)
                sender.send(pb_utils.InferenceResponse(
                    output_tensors=[audio_tensor]
                ))
                force_flush = False

            prev_num_frames = cur_num_frames
            with self._session_state_lock:
                self._session_prev_frames[seq_id] = prev_num_frames
                self._session_audio_buffers[seq_id] = audio_buf

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
