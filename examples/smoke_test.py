"""V0.0.1 Gemma-4-E2B NLA smoke test — fully self-contained, no source-repo dependency.

Validates a fresh-clone replicator's environment by:
  1. Loading the v0.0.1 AV+AR adapters from HuggingFace.
  2. Downloading the 20-row smoke-eval dataset from HuggingFace
     (Solshine/gemma-4-e2b-nla-eval-smoke).
  3. Running round-trip inference on the first 3 fixed activations.
  4. Asserting at least 2 of 3 clear the 0.30 noise floor.

Expected wall time: ~3-8 minutes (first run downloads ~5 GB of HF artifacts;
subsequent runs are cached).
Expected output: SMOKE TEST PASSED. cos values: [...]; N/3 above noise floor 0.30.

Run from the bundled repo root after `pip install -r requirements.txt`:
    python examples/smoke_test.py

If anything fails the assertion and traceback will tell you which step broke
(CUDA missing, Gemma license not accepted, OOM, etc.).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import snapshot_download, hf_hub_download

BASE_MODEL = "google/gemma-4-E2B"
AV_REPO = "Solshine/gemma-4-e2b-nla-L23-av-v0_0_1"
AR_REPO = "Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1"
SMOKE_DATASET = "Solshine/gemma-4-e2b-nla-eval-smoke"
SMOKE_PARQUET_NAME = "smoke_eval.parquet"
N_SMOKE_ROWS = 3
COS_FLOOR = 0.30
LAYER = 23
D_MODEL = 1536
AR_TRUNCATION = 18
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
    )


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


def main() -> int:
    print("[smoke] step 1/6: CUDA / GPU check")
    assert torch.cuda.is_available(), "CUDA is not available. NLA inference needs a CUDA GPU."
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  free VRAM: {free_gb:.2f} GB")
    assert free_gb >= 1.5, f"Need >= 1.5 GB free VRAM, got {free_gb:.2f} GB"

    print(f"[smoke] step 2/6: download {N_SMOKE_ROWS}-row smoke eval from {SMOKE_DATASET}")
    parquet_path = hf_hub_download(
        repo_id=SMOKE_DATASET, filename=SMOKE_PARQUET_NAME, repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()[:N_SMOKE_ROWS]
    print(f"  loaded {len(rows)} eval rows")

    print(f"[smoke] step 3/6: load AV from {AV_REPO}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=make_bnb_config(),
        device_map={"": torch.cuda.current_device()},
    )
    av = PeftModel.from_pretrained(base, AV_REPO)
    av.eval()
    pending = {"input_ids": None, "vec": None}
    embed_layer = av.get_input_embeddings()
    av_handle = embed_layer.register_forward_hook(make_av_inject_hook(pending))

    print(f"[smoke] step 4/6: load AR from {AR_REPO}")
    ar_local = Path(snapshot_download(repo_id=AR_REPO))
    base2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=make_bnb_config(),
        device_map={"": torch.cuda.current_device()},
    )
    ar = PeftModel.from_pretrained(base2, str(ar_local))
    ar.eval()
    head = nn.Linear(D_MODEL, D_MODEL, bias=True).to(ar.device).to(torch.float32)
    head.load_state_dict(torch.load(ar_local / "linear_head.pt"))
    head.eval()
    extraction_layer = ar.base_model.model.model.language_model.layers[AR_TRUNCATION - 1]

    print(f"[smoke] step 5/6: round-trip {N_SMOKE_ROWS} rows")
    coses = []
    try:
        for i, r in enumerate(rows):
            vec = np.array(r["activation_vector"], dtype=np.float32)
            norm = float(np.linalg.norm(vec)) + 1e-9
            vec_scaled = (vec / norm * float(np.sqrt(D_MODEL))).astype(np.float32)

            av_prompt = AV_TEMPLATE.format(injection_char=INJECTION_CHAR)
            ids = tok.encode(av_prompt, return_tensors="pt").to(av.device)
            pending["input_ids"] = ids
            pending["vec"] = torch.from_numpy(vec_scaled).to(av.device).unsqueeze(0)
            with torch.no_grad():
                out = av.generate(
                    input_ids=ids, max_new_tokens=120,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            pending["input_ids"] = None
            pending["vec"] = None
            explanation = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

            if not explanation.strip():
                print(f"  row {i}: empty AV output (the documented 16% small-model failure mode)")
                coses.append(float("nan"))
                continue

            ar_prompt = AR_TEMPLATE.format(explanation=explanation)
            ar_ids = tok.encode(ar_prompt, return_tensors="pt").to(ar.device)
            captured = {"h": None}

            def ar_hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                captured["h"] = h

            ar_handle = extraction_layer.register_forward_hook(ar_hook)
            try:
                with torch.no_grad():
                    _ = ar(input_ids=ar_ids)
            finally:
                ar_handle.remove()
            h_last = captured["h"][0, -1].to(torch.float32)
            recon = head(h_last).detach().cpu().numpy().astype(np.float32)

            v_n = vec / (np.linalg.norm(vec) + 1e-9)
            r_n = recon / (np.linalg.norm(recon) + 1e-9)
            cos = float(np.dot(v_n, r_n))
            coses.append(cos)
            print(f"  row {i}: cos={cos:.4f}  explanation[:80]={explanation[:80]!r}")
    finally:
        av_handle.remove()

    print(f"[smoke] step 6/6: assert >= 2 of {N_SMOKE_ROWS} rows clear cos > {COS_FLOOR}")
    above_floor = sum(1 for c in coses if not np.isnan(c) and c > COS_FLOOR)
    assert above_floor >= 2, (
        f"SMOKE TEST FAILED. Only {above_floor}/{N_SMOKE_ROWS} rows cleared cos > {COS_FLOOR}. "
        f"cos values: {coses}. Check: 1) base model gating (huggingface-cli login + accept Gemma license), "
        f"2) bitsandbytes version (must be 0.49.2), "
        f"3) device_map (integer form, not the string 'cuda')."
    )
    print(
        f"\nSMOKE TEST PASSED. cos values: {[round(c, 4) if not np.isnan(c) else None for c in coses]}; "
        f"{above_floor}/{N_SMOKE_ROWS} above noise floor {COS_FLOOR}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
