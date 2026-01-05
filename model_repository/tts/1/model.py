import json
import os
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from tts_generator import TTSGenerator


@dataclass
class TTSConfig:
    temperature: float = 0.01
    top_p: float = 0.999
    decoder_temperature: float = 0.01
    decoder_top_p: float = 0.999
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
            self._sent_audio_len: Dict[int, int] = {}

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
            self._sent_audio_len[seq_id] = 0
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Cache initialized successfully")
            return

        # Handles ending. Ends session
        if is_end:
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Session successfully ended!")
            self.tts_generator.end_session(seq_id)
            self._sent_audio_len.pop(seq_id, None)
            self._send_final(sender)
            return

        pb_utils.Logger.log_info(f"Appends intermediate texts: {texts}")
        ok = self.tts_generator.append_texts(seq_id, texts, speaker_id=0)
        if not ok:
            self._send_final(sender)
            raise RuntimeError(f"[Seq {seq_id}] session is not active!")

        # Processes one word audio
        # If model loops and never returns eos token, max_steps will stop it!
        sent_audio_len = self._sent_audio_len.get(seq_id, 0)
        pending_audio = np.zeros((0,), dtype=np.float32)
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

            # Decode full audio from all codes
            audio_24k = self.tts_generator._decode_audio(codes)
            audio_24k = audio_24k.cpu().float()

            audio_np = audio_24k.numpy()

            # Chunk length = 1920 @ 24k scaled by target_sr
            chunk_len = int(self.stream_chunk_duration * float(self.output_sample_rate))

            if audio_np.size <= sent_audio_len:
                if is_complete:
                    break
                continue

            delta = audio_np[sent_audio_len:]
            sent_audio_len = audio_np.size
            if delta.size > 0:
                pending_audio = (
                    delta.astype(np.float32)
                    if pending_audio.size == 0
                    else np.concatenate([pending_audio, delta.astype(np.float32)])
                )

            while pending_audio.size >= chunk_len:
                chunk_np = pending_audio[:chunk_len].astype(np.float32)
                pending_audio = pending_audio[chunk_len:]
                audio_tensor = pb_utils.Tensor("AUDIO_FRAME", chunk_np)
                sender.send(pb_utils.InferenceResponse(
                    output_tensors=[audio_tensor]
                ))

            # Means model returned eos token
            if is_complete:
                break

            # Means max_steps stopped the process. Because of that, we must increase word pointer
            if cur_step == self.config.max_steps - 1:
                self.tts_generator.increment_state_counter(seq_id)

        if pending_audio.size > 0:
            audio_tensor = pb_utils.Tensor("AUDIO_FRAME", pending_audio.astype(np.float32))
            sender.send(pb_utils.InferenceResponse(
                output_tensors=[audio_tensor]
            ))

        self._sent_audio_len[seq_id] = sent_audio_len

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
