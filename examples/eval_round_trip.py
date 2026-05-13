"""V8 Eval — round-trip reconstruction MSE on held-out (rl) parquet.

Load AV + AR, run held-out activations through:
  vector -> AV(text) -> AR(reconstructed_vector)
and compute MSE between original and reconstructed (both L2-normalized to sqrt_d_model).

Compare to:
  - Random baseline: MSE between random unit vectors (≈ 2.0 for unit vecs)
  - Identity baseline: AR called on a literal repeat of the input prompt

Usage:
    python eval_round_trip.py \\
        --av-checkpoint checkpoints/av_sft_v0/final/ \\
        --ar-checkpoint checkpoints/ar_sft_v0/final/ \\
        --eval-data data/stage1/rl.parquet \\
        --output results/round_trip_v0.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import pyarrow.parquet as pq

D_MODEL = 1536
INJECTION_TOKEN_ID = 249568
INJECTION_LEFT_NEIGHBOR_ID = 236813
INJECTION_RIGHT_NEIGHBOR_ID = 954
INJECTION_CHAR = chr(0x3297)


def load_av(checkpoint_path: Path, base_model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                  device_map={"": torch.cuda.current_device()})
    av = PeftModel.from_pretrained(base, str(checkpoint_path))
    av.eval()
    return av, tok


def load_ar(checkpoint_path: Path, base_model: str, ar_truncation: int = 18):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                  device_map={"": torch.cuda.current_device()})
    ar = PeftModel.from_pretrained(base, str(checkpoint_path))
    head = nn.Linear(D_MODEL, D_MODEL, bias=True).to(ar.device).to(torch.float32)
    head.load_state_dict(torch.load(checkpoint_path / "linear_head.pt"))
    head.eval()
    ar.eval()

    layer_modules = ar.base_model.model.model.language_model.layers
    extraction_layer = layer_modules[ar_truncation - 1]
    return ar, tok, head, extraction_layer


AV_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>"""


def av_explain(av, tok, vec: np.ndarray, max_new_tokens: int = 120,
                pending: dict | None = None) -> str:
    """Run AV with forward-hook injection (avoids Gemma 4 inputs_embeds OOM).

    pending: shared dict that the embedding-layer hook reads from.
       Caller must register the hook once (see main()); we just toggle the
       payload via pending["input_ids"], pending["vec"].
    """
    prompt = AV_TEMPLATE.format(injection_char=INJECTION_CHAR)
    ids = tok.encode(prompt, return_tensors="pt").to(av.device)

    norm = float(np.linalg.norm(vec)) + 1e-9
    scaled = (vec / norm * float(np.sqrt(D_MODEL))).astype(np.float32)
    vec_t = torch.from_numpy(scaled).to(av.device).unsqueeze(0)  # [1, d_model]

    if pending is None:
        return ""
    pending["input_ids"] = ids
    pending["vec"] = vec_t
    try:
        with torch.no_grad():
            out = av.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        pending["input_ids"] = None
        pending["vec"] = None
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return text


def make_av_inject_hook(pending: dict):
    """Returns a forward-hook for embed_layer that injects pending vec at marker trio."""
    def hook(module, args_in, output):
        # Only fire during prompt encoding (output.shape[1] > 1); skip incremental decode.
        if output.shape[1] <= 1: return output
        ids = pending.get("input_ids"); vec = pending.get("vec")
        if ids is None or vec is None: return output
        h = output.clone()
        for b in range(min(ids.shape[0], h.shape[0])):
            for p2 in range(1, min(ids.shape[1], h.shape[1]) - 1):
                if (ids[b, p2].item() == INJECTION_TOKEN_ID
                    and ids[b, p2-1].item() == INJECTION_LEFT_NEIGHBOR_ID
                    and ids[b, p2+1].item() == INJECTION_RIGHT_NEIGHBOR_ID):
                    h[b, p2] = vec[b].to(h.dtype)
                    break
        return h
    return hook


def ar_reconstruct(ar, tok, head, ext_layer, explanation: str) -> np.ndarray:
    """Run AR: text -> vector via Linear head at last token."""
    AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
    prompt = AR_TEMPLATE.format(explanation=explanation)
    ids = tok.encode(prompt, return_tensors="pt").to(ar.device)
    captured = {"h": None}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        captured["h"] = h
    handle = ext_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = ar(input_ids=ids)
    finally:
        handle.remove()
    if captured["h"] is None: return np.zeros(D_MODEL, dtype=np.float32)
    h_last = captured["h"][0, -1].to(torch.float32)
    recon = head(h_last)
    return recon.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--av-checkpoint", type=Path, required=True)
    p.add_argument("--ar-checkpoint", type=Path, required=True)
    p.add_argument("--eval-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model", default="google/gemma-4-E2B")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=120)
    args = p.parse_args()

    print(f"[V8 eval] loading AV from {args.av_checkpoint}")
    av, tok = load_av(args.av_checkpoint, args.base_model)
    print(f"[V8 eval] loading AR from {args.ar_checkpoint}")
    ar, _, head, ext_layer = load_ar(args.ar_checkpoint, args.base_model)

    pending = {"input_ids": None, "vec": None}
    embed_layer = av.get_input_embeddings()
    handle = embed_layer.register_forward_hook(make_av_inject_hook(pending))

    eval_table = pq.read_table(args.eval_data)
    rows = eval_table.to_pylist()[:args.limit]
    print(f"[V8 eval] running on {len(rows)} held-out activations")

    results = []
    try:
        for i, r in enumerate(tqdm(rows)):
            vec = np.array(r["activation_vector"], dtype=np.float32)
            # Round-trip
            explanation = av_explain(av, tok, vec, max_new_tokens=args.max_new_tokens, pending=pending)
            if not explanation: continue
            recon = ar_reconstruct(ar, tok, head, ext_layer, explanation)

            # Normalize both to direction-only
            v_n = vec / (np.linalg.norm(vec) + 1e-9)
            r_n = recon / (np.linalg.norm(recon) + 1e-9)
            cos = float(np.dot(v_n, r_n))
            mse = float(np.sum((v_n - r_n) ** 2))
            results.append({
                "doc_id": r["doc_id"],
                "n_raw_tokens": r["n_raw_tokens"],
                "round_trip_cos": cos, "round_trip_mse": mse,
                "explanation": explanation[:300],
            })
    finally:
        handle.remove()

    cos_vals = [r["round_trip_cos"] for r in results]
    mse_vals = [r["round_trip_mse"] for r in results]
    summary = {
        "n_evaluated": len(results),
        "round_trip_mse": {
            "mean": float(np.mean(mse_vals)) if mse_vals else float("nan"),
            "median": float(np.median(mse_vals)) if mse_vals else float("nan"),
            "std": float(np.std(mse_vals)) if mse_vals else float("nan"),
        },
        "round_trip_cos": {
            "mean": float(np.mean(cos_vals)) if cos_vals else float("nan"),
            "median": float(np.median(cos_vals)) if cos_vals else float("nan"),
            "std": float(np.std(cos_vals)) if cos_vals else float("nan"),
        },
        "random_baseline_mse": 2.0,  # for reference
    }
    print(f"\nSummary: {summary}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_row": results}, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
