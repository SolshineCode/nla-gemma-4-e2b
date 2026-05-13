"""Self-contained round-trip inference example for the v0.0.1 Gemma-4-E2B NLA pair.

Takes a string of text from the command line, extracts a layer-23 residual-stream
activation from `google/gemma-4-E2B` at the LAST token of that text, runs the
activation through the AV (Actor) to produce a natural-language explanation,
runs the explanation through the AR (Critic) to reconstruct an activation,
and prints the round-trip cosine similarity.

No source-repo dependency. No local data files required. All artifacts pulled
from HuggingFace. First run downloads ~5 GB of base + adapter weights;
subsequent runs are cached.

Usage:
    python examples/round_trip_example.py "your input text here"
    python examples/round_trip_example.py "The cat sat on the mat."

Expected runtime: ~3-5 minutes on a 4 GB GPU. ~30 seconds on an A100.

Prerequisites:
    pip install -r requirements.txt
    huggingface-cli login   # paste your HF token; accept the Gemma-4-E2B license
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import snapshot_download

BASE_MODEL = "google/gemma-4-E2B"
AV_REPO = "Solshine/gemma-4-e2b-nla-L23-av-v0_0_1"
AR_REPO = "Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1"
LAYER = 23
D_MODEL = 1536
AR_TRUNCATION = 18
# Special tokens used to mark the injection slot in the AV prompt.
INJECTION_TOKEN_ID = 249568
INJECTION_LEFT_NEIGHBOR_ID = 236813
INJECTION_RIGHT_NEIGHBOR_ID = 954
INJECTION_CHAR = chr(0x3297)
AV_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>"""
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def extract_activation(text: str) -> tuple[np.ndarray, str]:
    """Load Gemma-4-E2B, run forward pass on `text`, return the L23 activation
    at the last token (L2-normalized to sqrt(d_model)) and the detokenized
    text snippet."""
    print(f"[1/4] Loading {BASE_MODEL} in 4-bit NF4 for activation extraction...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=make_bnb_config(),
        device_map={"": torch.cuda.current_device()},
    )
    base.eval()
    layer_module = base.model.language_model.layers[LAYER]
    captured = {"acts": None}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["acts"] = h.detach().cpu().float()

    handle = layer_module.register_forward_hook(hook)
    ids = tok.encode(text, return_tensors="pt").to(base.device)
    try:
        with torch.no_grad():
            base(input_ids=ids)
    finally:
        handle.remove()

    h = captured["acts"][0]  # [seq, d_model]
    vec = h[-1].numpy().astype(np.float32)  # last-token activation
    norm = float(np.linalg.norm(vec)) + 1e-9
    vec_scaled = (vec / norm * float(np.sqrt(D_MODEL))).astype(np.float32)
    detok = tok.decode(ids[0], skip_special_tokens=True)

    # Free Gemma-4-E2B before loading AV + AR
    del base
    torch.cuda.empty_cache()
    return vec_scaled, detok


def make_av_inject_hook(pending: dict):
    def hook(module, args_in, output):
        if output.shape[1] <= 1:
            return output
        ids = pending.get("input_ids")
        vec = pending.get("vec")
        if ids is None or vec is None:
            return output
        h = output.clone()
        for b in range(min(ids.shape[0], h.shape[0])):
            for p2 in range(1, min(ids.shape[1], h.shape[1]) - 1):
                if (ids[b, p2].item() == INJECTION_TOKEN_ID
                    and ids[b, p2 - 1].item() == INJECTION_LEFT_NEIGHBOR_ID
                    and ids[b, p2 + 1].item() == INJECTION_RIGHT_NEIGHBOR_ID):
                    h[b, p2] = vec[b].to(h.dtype)
                    break
        return h
    return hook


def av_explain(vec_scaled: np.ndarray) -> str:
    print(f"[2/4] Loading AV adapter from {AV_REPO}...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=make_bnb_config(),
        device_map={"": torch.cuda.current_device()},
    )
    av = PeftModel.from_pretrained(base, AV_REPO)
    av.eval()

    prompt = AV_TEMPLATE.format(injection_char=INJECTION_CHAR)
    ids = tok.encode(prompt, return_tensors="pt").to(av.device)
    vec_t = torch.from_numpy(vec_scaled).to(av.device).unsqueeze(0)

    pending = {"input_ids": ids, "vec": vec_t}
    embed_layer = av.get_input_embeddings()
    handle = embed_layer.register_forward_hook(make_av_inject_hook(pending))
    try:
        with torch.no_grad():
            out = av.generate(
                input_ids=ids, max_new_tokens=120,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
    finally:
        handle.remove()
    explanation = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    del av, base
    torch.cuda.empty_cache()
    return explanation


def ar_reconstruct(explanation: str) -> np.ndarray:
    print(f"[3/4] Loading AR adapter from {AR_REPO}...")
    ar_local = Path(snapshot_download(repo_id=AR_REPO))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=make_bnb_config(),
        device_map={"": torch.cuda.current_device()},
    )
    ar = PeftModel.from_pretrained(base, str(ar_local))
    ar.eval()
    head = nn.Linear(D_MODEL, D_MODEL, bias=True).to(ar.device).to(torch.float32)
    head.load_state_dict(torch.load(ar_local / "linear_head.pt"))
    head.eval()

    layer_modules = ar.base_model.model.model.language_model.layers
    extraction_layer = layer_modules[AR_TRUNCATION - 1]

    prompt = AR_TEMPLATE.format(explanation=explanation)
    ids = tok.encode(prompt, return_tensors="pt").to(ar.device)
    captured = {"h": None}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        captured["h"] = h

    handle = extraction_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = ar(input_ids=ids)
    finally:
        handle.remove()

    h_last = captured["h"][0, -1].to(torch.float32)
    recon = head(h_last)
    return recon.detach().cpu().numpy().astype(np.float32)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python examples/round_trip_example.py \"your input text here\"")
        return 2
    text = sys.argv[1]

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script needs an NVIDIA GPU with >= 1.5 GB free VRAM.")
        return 1
    free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({free_gb:.2f} GB free)")
    print()
    print(f"Input text: {text!r}")
    print()

    vec_original, detok = extract_activation(text)
    explanation = av_explain(vec_original)
    print()
    print(f"[AV explanation]")
    print(explanation)
    print()

    vec_reconstructed = ar_reconstruct(explanation)
    print()
    print("[4/4] Computing round-trip cosine similarity")
    v_n = vec_original / (np.linalg.norm(vec_original) + 1e-9)
    r_n = vec_reconstructed / (np.linalg.norm(vec_reconstructed) + 1e-9)
    cos = float(np.dot(v_n, r_n))
    print()
    print(f"Round-trip cosine similarity: {cos:.4f}")
    print(f"  - above 0.30 noise floor: {'YES' if cos > 0.30 else 'NO'}")
    print(f"  - v0.0.1 published mean on n=42: 0.438")
    print()
    print("READ THIS BEFORE INTERPRETING THE RESULT:")
    print("  At v0.0.1 scale, the AV converges to ~20 unique strings across our 42-row eval,")
    print("  4 opening genre stems (legal case / protest / new feature / new policy), 52%")
    print("  exact-duplicate rate. The explanation above is likely one of those templates")
    print("  rather than a faithful per-row description of YOUR activation. Round-trip cos")
    print("  is the pair's system-level closure, not a per-row interpretability claim.")
    print("  See the 'AV template collapse' section of the model card and the full")
    print("  ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md write-up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
