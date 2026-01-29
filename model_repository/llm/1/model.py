"""
LLM Model for Triton Inference Server with Streaming Support.
Uses decoupled mode to stream tokens one at a time.
"""

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import json
import time


class TritonPythonModel:

    def initialize(self, args):
        """
        Initialize the LLM model.
        """
        # Get model path from parameters or use default
        model_config = json.loads(args['model_config'])
        
        # Default model path - LLM weights are mounted at /llm_weights
        model_path = "/local_models/llm_model"
        
        # Check for custom path in parameters
        if 'parameters' in model_config:
            params = model_config['parameters']
            if 'model_path' in params:
                model_path = params['model_path']['string_value']
        
        pb_utils.Logger.log_info(f"Loading LLM from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with 8-bit quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        
        # Default generation parameters
        self.default_max_tokens = 512
        self.default_temperature = 0.7
        self.default_top_p = 0.9
        
        pb_utils.Logger.log_info("LLM model initialized successfully with streaming support")

    def execute(self, requests):
        """
        Execute streaming inference using decoupled mode.
        Streams tokens one at a time to the client.
        """
        for request in requests:
            response_sender = request.get_response_sender()
            
            try:
                # Get input prompt
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                prompt = prompt_tensor.as_numpy()[0]
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")
                
                # Get optional parameters
                max_new_tokens = self.default_max_tokens
                temperature = self.default_temperature
                top_p = self.default_top_p
                
                max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "MAX_NEW_TOKENS")
                if max_tokens_tensor is not None:
                    max_new_tokens = int(max_tokens_tensor.as_numpy()[0])
                
                temp_tensor = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")
                if temp_tensor is not None:
                    temperature = float(temp_tensor.as_numpy()[0])
                
                top_p_tensor = pb_utils.get_input_tensor_by_name(request, "TOP_P")
                if top_p_tensor is not None:
                    top_p = float(top_p_tensor.as_numpy()[0])
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(self.model.device)
                
                # Create streamer
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                # Generation kwargs
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature if temperature > 0 else 1.0,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "streamer": streamer,
                }
                
                # Start generation in a thread
                start = time.perf_counter()
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Stream tokens
                chunks: list[str] = []
                for text_chunk in streamer:
                    if text_chunk:
                        chunks.append(text_chunk)
                        chunk_tensor = pb_utils.Tensor(
                            "TEXT_CHUNK",
                            np.array([text_chunk.encode("utf-8")], dtype=object)
                        )
                        finished_tensor = pb_utils.Tensor(
                            "FINISHED",
                            np.array([False], dtype=bool)
                        )
                        response = pb_utils.InferenceResponse([chunk_tensor, finished_tensor])
                        response_sender.send(response)
                
                # Wait for thread to finish
                thread.join()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                full_text = "".join(chunks)
                token_count = len(self.tokenizer.encode(full_text, add_special_tokens=False)) if full_text else 0
                tok_per_s = (token_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
                
                # Send final response
                chunk_tensor = pb_utils.Tensor(
                    "TEXT_CHUNK",
                    np.array(["".encode("utf-8")], dtype=object)
                )
                finished_tensor = pb_utils.Tensor(
                    "FINISHED",
                    np.array([True], dtype=bool)
                )
                final_response = pb_utils.InferenceResponse([chunk_tensor, finished_tensor])
                response_sender.send(
                    final_response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                
                pb_utils.Logger.log_info(
                    "LLM | total_ms=%.1f | tokens=%d | tok_per_s=%.2f"
                    % (elapsed_ms, token_count, tok_per_s)
                )
                
            except Exception as e:
                pb_utils.Logger.log_error(f"Error in LLM streaming: {str(e)}")
                import traceback
                pb_utils.Logger.log_error(traceback.format_exc())
                
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )
                response_sender.send(
                    error_response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        
        return None  # Decoupled mode - responses sent via response_sender

    def finalize(self):
        """
        Clean up resources.
        """
        pb_utils.Logger.log_info("LLM model finalized")
