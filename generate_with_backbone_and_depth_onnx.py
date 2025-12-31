#!/usr/bin/env python3
# generate_with_backbone_and_depth_onnx.py

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort

from transformers import CsmForConditionalGeneration, AutoProcessor

from export_depth_decoder_onnx import (
    AUDIO_CONTEXT_PATH,
    BACKBONE_ONNX_PATH,
    BACKBONE_PAST_LEN,
    DEPTH_DECODER_ONNX_PATH,
    DEPTH_DECODER_PAST_LEN,
    JSON_CONTEXT_PATH,
    MODEL_PATH,
    PROCESSOR_PATH,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def sample_from_logits(logits, method="greedy", temperature=1.0, top_k=0, top_p=1.0):
    if method == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)
    temperature = max(float(temperature), 1e-8)
    probs = F.softmax(logits / temperature, dim=-1)

    if method == "multinomial":
        return torch.multinomial(probs, num_samples=1)

    if method == "topk":
        k = int(top_k)
        topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(topk_probs, num_samples=1)
        return topk_idx.gather(-1, sampled)

    if method == "nucleus":
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)
        mask = cumprobs <= float(top_p)
        mask[..., 0] = True
        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_idx.gather(-1, sampled)

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def generate_audio_from_frames(codec_model, codebook_eos_token_id, device, frames_list):
    x = torch.stack(frames_list, dim=1).to(device=device, dtype=torch.long)  # [B,T,32]

    eos_id = int(codebook_eos_token_id)
    eos_mask = (x == eos_id).all(dim=-1)

    audios = []
    for b in range(x.shape[0]):
        if eos_mask[b].any():
            cutoff = int(torch.nonzero(eos_mask[b], as_tuple=False)[0].item())
        else:
            cutoff = x.shape[1]
        xb = x[b, :cutoff, :]
        codec_in = xb.transpose(0, 1).unsqueeze(0)  # [1,32,T]
        out = codec_model.decode(codec_in)
        audios.append(out.audio_values[0, 0])
    return audios


def _pick_providers(device_id: int):
    provs = ort.get_available_providers()
    torch_stream = torch.cuda.current_stream(device_id).cuda_stream

    providers = []
    # If you have ORT built with TensorRT EP, this is the only realistic way to approach PyTorch SDPA speed.
    if "TensorrtExecutionProvider" in provs:
        providers.append(("TensorrtExecutionProvider", {"device_id": device_id}))
    providers.append((
        "CUDAExecutionProvider",
        {"device_id": device_id, "user_compute_stream": torch_stream},
    ))
    providers.append("CPUExecutionProvider")
    return providers


class ORTDepthDecoder:
    def __init__(self, onnx_path: str, past_len: int, device_id: int = 0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=_pick_providers(device_id),
        )

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        self.L = (len(ins) - 4) // 2
        B, KVH, P, HD = map(int, ins[4].shape)
        assert P == past_len

        B2, S2, V = map(int, outs[0].shape)
        assert B2 == B and S2 == 1

        B3, Hbb = map(int, ins[1].shape)
        assert B3 == B

        self.B, self.KVH, self.HD, self.V, self.Hbb = B, KVH, HD, V, Hbb
        self.past_len = past_len

        self.token_buf = torch.zeros((B, 1), device="cuda", dtype=torch.long)
        self.bb_buf = torch.zeros((B, Hbb), device="cuda", dtype=torch.float16)
        self.cache_pos = torch.zeros((1,), device="cuda", dtype=torch.long)
        self.mask_buf = torch.zeros((B, 1, 1, past_len + 1), device="cuda", dtype=torch.float16)

        self.past = [torch.zeros((B, KVH, past_len, HD), device="cuda", dtype=torch.float16) for _ in range(2 * self.L)]
        self.present = [torch.empty((B, KVH, past_len + 1, HD), device="cuda", dtype=torch.float16) for _ in range(2 * self.L)]
        self.out_logits = torch.empty((B, 1, V), device="cuda", dtype=torch.float16)

        self.io = self.sess.io_binding()
        self.io.bind_input("input_ids", "cuda", 0, np.int64, self.token_buf.shape, self.token_buf.data_ptr())
        self.io.bind_input("backbone_last_hidden_state", "cuda", 0, np.float16, self.bb_buf.shape, self.bb_buf.data_ptr())
        self.io.bind_input("attention_mask", "cuda", 0, np.float16, self.mask_buf.shape, self.mask_buf.data_ptr())
        self.io.bind_input("cache_position", "cuda", 0, np.int64, self.cache_pos.shape, self.cache_pos.data_ptr())

        for i in range(2 * self.L):
            t = self.past[i]
            self.io.bind_input(f"past_{i}", "cuda", 0, np.float16, t.shape, t.data_ptr())

        self.io.bind_output("logits", "cuda", 0, np.float16, self.out_logits.shape, self.out_logits.data_ptr())
        for i in range(2 * self.L):
            t = self.present[i]
            self.io.bind_output(f"present_{i}", "cuda", 0, np.float16, t.shape, t.data_ptr())

    @torch.no_grad()
    def _set_mask_for_pos(self, pos: int):
        valid = min(int(pos), self.past_len)
        start = self.past_len - valid
        self.mask_buf.zero_()
        if start > 0:
            self.mask_buf[..., :start] = -1e4

    @torch.no_grad()
    def reset(self):
        for t in self.past:
            t.zero_()

    @torch.no_grad()
    def step(self, token_id: torch.Tensor, h_last: torch.Tensor, pos: int) -> torch.Tensor:
        if token_id.dim() == 1:
            token_id = token_id.view(self.B, 1)
        self.token_buf.copy_(token_id)
        self.bb_buf.copy_(h_last.to(torch.float16))
        self.cache_pos.fill_(int(pos))
        self._set_mask_for_pos(pos)

        self.sess.run_with_iobinding(self.io)

        for i in range(2 * self.L):
            self.past[i].copy_(self.present[i][:, :, 1:, :])

        return self.out_logits[:, 0, :]

    @torch.no_grad()
    def generate_frame(self, c0: torch.Tensor, h_last: torch.Tensor, method="nucleus", temperature=0.1, top_p=0.999):
        B = self.B
        if c0.dim() == 2:
            c0 = c0[:, 0]
        c0 = c0.to(torch.long).view(B, 1)

        self.reset()

        dummy = torch.zeros((B, 1), device="cuda", dtype=torch.long)
        _ = self.step(dummy, h_last, pos=0)

        logits = self.step(c0, h_last, pos=1)
        c1 = sample_from_logits(logits, method=method, temperature=temperature, top_p=top_p)

        outs = [c1]
        prev = c1
        for pos in range(2, 32):
            logits = self.step(prev, h_last, pos=pos)
            nxt = sample_from_logits(logits, method=method, temperature=temperature, top_p=top_p)
            outs.append(nxt)
            prev = nxt

        rest = torch.cat(outs, dim=1)
        return torch.cat([c0, rest], dim=1)


class ORTBackbonePastN:
    def __init__(self, onnx_path: str, past_len: int, device_id: int = 0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=_pick_providers(device_id),
        )

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        self.L = (len(ins) - 3) // 2

        B, KVH, P, HD = map(int, ins[3].shape)
        assert P == past_len

        B2, S2, H = map(int, ins[0].shape)
        assert B2 == B and S2 == 1

        B3, S3, V = map(int, outs[0].shape)
        assert B3 == B and S3 == 1

        B4, H2 = map(int, outs[1].shape)
        assert B4 == B and H2 == H

        # new_0 exists at outs[2]
        B5, KVH2, ONE, HD2 = map(int, outs[2].shape)
        assert B5 == B and KVH2 == KVH and ONE == 1 and HD2 == HD

        self.B, self.KVH, self.HD, self.V, self.H = B, KVH, HD, V, H
        self.past_len = past_len

        self.embed_buf = torch.zeros((B, 1, H), device="cuda", dtype=torch.float16)
        self.mask_buf = torch.zeros((B, 1, 1, past_len + 1), device="cuda", dtype=torch.float16)
        self.cache_pos = torch.zeros((1,), device="cuda", dtype=torch.long)

        self.past = [torch.zeros((B, KVH, past_len, HD), device="cuda", dtype=torch.float16) for _ in range(2 * self.L)]
        self.out_logits = torch.empty((B, 1, V), device="cuda", dtype=torch.float16)
        self.out_hidden = torch.empty((B, H), device="cuda", dtype=torch.float16)
        self.new_kv = [torch.empty((B, KVH, 1, HD), device="cuda", dtype=torch.float16) for _ in range(2 * self.L)]

        self.io = self.sess.io_binding()
        self.io.bind_input("inputs_embeds", "cuda", 0, np.float16, self.embed_buf.shape, self.embed_buf.data_ptr())
        self.io.bind_input("attention_mask", "cuda", 0, np.float16, self.mask_buf.shape, self.mask_buf.data_ptr())
        self.io.bind_input("cache_position", "cuda", 0, np.int64, self.cache_pos.shape, self.cache_pos.data_ptr())

        for i in range(2 * self.L):
            t = self.past[i]
            self.io.bind_input(f"past_{i}", "cuda", 0, np.float16, t.shape, t.data_ptr())

        self.io.bind_output("logits", "cuda", 0, np.float16, self.out_logits.shape, self.out_logits.data_ptr())
        self.io.bind_output("last_hidden_state", "cuda", 0, np.float16, self.out_hidden.shape, self.out_hidden.data_ptr())

        for i in range(2 * self.L):
            t = self.new_kv[i]
            self.io.bind_output(f"new_{i}", "cuda", 0, np.float16, t.shape, t.data_ptr())

        self._all_valid_mask = False
        self.mask_buf.zero_()  # default for warm state once it becomes all-valid

    @torch.no_grad()
    def reset(self):
        for t in self.past:
            t.zero_()
        self.mask_buf.zero_()
        self._all_valid_mask = False

    @torch.no_grad()
    def _set_mask_for_pos(self, pos: int):
        pos = int(pos)
        if pos >= self.past_len:
            if not self._all_valid_mask:
                self.mask_buf.zero_()
                self._all_valid_mask = True
            return

        # early steps only
        self._all_valid_mask = False
        valid = pos
        self.mask_buf.zero_()
        if valid < self.past_len:
            self.mask_buf[..., valid:self.past_len] = -1e4  # do not touch last slot (current)

    @torch.no_grad()
    def step(self, inputs_embeds: torch.Tensor, pos: int):
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(1)
        # expect fp16 already
        self.embed_buf.copy_(inputs_embeds)
        self.cache_pos.fill_(int(pos))
        self._set_mask_for_pos(pos)

        self.sess.run_with_iobinding(self.io)

        idx = int(pos) % self.past_len
        for i in range(2 * self.L):
            self.past[i][:, :, idx:idx + 1, :].copy_(self.new_kv[i])

        return self.out_logits, self.out_hidden


@torch.no_grad()
def prefill_backbone_with_onnx(ort_bb, inputs_embeds):
    seq_len = inputs_embeds.shape[1]
    logits_next = None
    h_last = None
    for i in range(seq_len):
        token_embed = inputs_embeds[:, i:i+1, :]
        ort_logits, h_last = ort_bb.step(token_embed, pos=i)
        logits_next = ort_logits[:, 0, :]
    return logits_next, h_last, seq_len


def main():
    device = "cuda"

    SAMPLE_METHOD = "nucleus"
    TEMPERATURE = 0.1
    TOP_P = 0.999
    MAX_STEPS = 125 // 2

    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)

    # Keep model mostly on CPU, but make embedding/codec fp16 on GPU to avoid casts
    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval()

    model.embed_text_tokens = model.embed_text_tokens.to(device=device, dtype=torch.float16)
    model.backbone_model.embed_tokens = model.backbone_model.embed_tokens.to(device=device, dtype=torch.float16)
    model.codec_model = model.codec_model.to(device)

    text_embed = model.embed_text_tokens
    audio_embed = model.backbone_model.embed_tokens
    codec_model = model.codec_model

    ort_dd = ORTDepthDecoder(DEPTH_DECODER_ONNX_PATH, past_len=DEPTH_DECODER_PAST_LEN, device_id=0)
    ort_bb = ORTBackbonePastN(BACKBONE_ONNX_PATH, past_len=BACKBONE_PAST_LEN, device_id=0)
    ort_bb.reset()

    whole_text = (
        "ქართველური ტომების გაერთიანების უმთავრეს მიზეზად ამ ტომთა საერთო საქმიანობა უნდა "
        "ჩავთვალოთ - სამთამადნო წარმოება, მეტალურგია და ლითონდამუშავება ."
    )
    words = whole_text.split()
    texts = [" ".join(words[:3])] + [" " + w for w in words[3:]] + ["", ""]
    print("texts:", texts)

    audio_ctx, _ = sf.read(AUDIO_CONTEXT_PATH, dtype="float32")
    with open(JSON_CONTEXT_PATH) as f:
        audio_transcript = json.load(f)

    speaker_id = 0

    all_inputs = []
    conversation = [
        {
            "role": f"{speaker_id}",
            "content": [
                {"type": "text", "text": audio_transcript[AUDIO_CONTEXT_PATH.split('/')[-1]]},
                {"type": "audio", "path": audio_ctx},
            ],
        },
        {"role": f"{speaker_id}", "content": [{"type": "text", "text": texts[0]}]},
    ]
    padded_inputs_1 = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(device)
    padded_inputs_1["input_values"] = padded_inputs_1["input_values"].to(torch.bfloat16)
    all_inputs.append(padded_inputs_1)

    for t in texts[1:]:
        conversation = [{"role": f"{speaker_id}", "content": [{"type": "text", "text": t}]}]
        cur_input = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(device)
        all_inputs.append(cur_input)

    counter = 0
    inputs = all_inputs[counter]

    model_inputs = model.prepare_inputs_for_generation(**inputs)
    inputs_embeds = model_inputs["inputs_embeds"].to(torch.float16)

    logits_next, h_last, seq_len = prefill_backbone_with_onnx(ort_bb, inputs_embeds)

    eos = torch.tensor([model.config.codebook_eos_token_id], device=device, dtype=torch.long)
    frames = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    cur_step = 0
    while cur_step < MAX_STEPS:
        c0 = sample_from_logits(logits_next, method=SAMPLE_METHOD, temperature=TEMPERATURE, top_p=TOP_P)
        c0_1d = c0.view(-1).to(torch.long)

        reached_eos = (c0_1d == eos.view(-1)).any()
        if bool(reached_eos) or cur_step == MAX_STEPS - 1:
            counter += 1
            if counter == len(all_inputs):
                break
            cur_step = 0
            inject_inputs = all_inputs[counter]
            inject_embeds = text_embed(inject_inputs["input_ids"])  # fp16
            for i in range(inject_embeds.shape[1]):
                token_embed = inject_embeds[:, i:i+1, :]
                ort_logits, h_last = ort_bb.step(token_embed, pos=seq_len)
                seq_len += 1
                logits_next = ort_logits[:, 0, :]
            continue

        frame = ort_dd.generate_frame(
            c0=c0_1d,
            h_last=h_last,  # fp16 ok (dd casts to fp16)
            method=SAMPLE_METHOD,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        frames.append(frame)

        frame_3d = frame.unsqueeze(1)
        audio_embeds = audio_embed(frame_3d)  # fp16
        ort_logits, h_last = ort_bb.step(audio_embeds, pos=seq_len)
        seq_len += 1
        logits_next = ort_logits[:, 0, :]

        cur_step += 1

    end.record()
    torch.cuda.synchronize()
    gen_time = start.elapsed_time(end) / 1000.0

    audio = generate_audio_from_frames(codec_model, model.config.codebook_eos_token_id, device, frames)[0]
    audio = audio.to(torch.float32).cpu()

    sr = 24_000
    audio_dur = len(audio) / sr
    print(f"Generation time: {gen_time:.3f} s")
    print(f"Audio duration: {audio_dur:.3f} s")
    print(f"RTF: {gen_time / max(audio_dur, 1e-9):.3f}")

    sf.write("output_onnx_bb_dd.wav", audio.numpy(), sr)
    print("Wrote output_onnx_bb_dd.wav")


if __name__ == "__main__":
    main()
