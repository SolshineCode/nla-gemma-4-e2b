---
license: cc-by-4.0
language:
  - en
base_model: google/gemma-4-E2B
tags:
  - natural-language-autoencoder
  - nla
  - interpretability
  - mechanistic-interpretability
  - gemma
  - consumer-gpu
  - peft
  - lora
library_name: peft
pipeline_tag: text-generation
---

# Gemma-4-E2B NLA AR (Activation Reconstructor) — v0.0.1

LoRA adapter (+ a 1536→1536 linear head) for `google/gemma-4-E2B` that takes a natural-language explanation produced by the matched [Activation Verbalizer](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1) and reconstructs a 1536-dimensional activation vector intended to round-trip (by cosine similarity) to the original L23 residual-stream activation.

Trained end-to-end on a single 4 GB consumer GPU (NVIDIA GTX 1650 Ti Max-Q) following the methodology of Fraser-Taliente, Kantamneni, Ong et al. 2026 ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/)).

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import snapshot_download
import torch
import torch.nn as nn
import numpy as np

BASE = "google/gemma-4-E2B"
AR_REPO = "Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1"
AR_TRUNCATION = 18   # capture hidden state at layer (K+1-1) = 17 (0-indexed)
D_MODEL = 1536
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
ar_local = snapshot_download(repo_id=AR_REPO)
ar = PeftModel.from_pretrained(base, ar_local); ar.eval()

# Load the trained linear head (1536 -> 1536)
head = nn.Linear(D_MODEL, D_MODEL, bias=True).to(ar.device).to(torch.float32)
head.load_state_dict(torch.load(f"{ar_local}/linear_head.pt", weights_only=True))
head.eval()

# Capture hidden state at layer L17 (1-indexed L18)
extraction_layer = ar.base_model.model.model.language_model.layers[AR_TRUNCATION - 1]

def reconstruct(explanation: str) -> np.ndarray:
    """Run an AV explanation through the AR; return the reconstructed activation vector."""
    prompt = AR_TEMPLATE.format(explanation=explanation)
    ids = tok.encode(prompt, return_tensors="pt").to(ar.device)
    captured = {"h": None}
    def hook(m, i, o):
        captured["h"] = o[0] if isinstance(o, tuple) else o
    handle = extraction_layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = ar(input_ids=ids)
    finally:
        handle.remove()
    h_last = captured["h"][0, -1].to(torch.float32)
    return head(h_last).detach().cpu().numpy().astype(np.float32)

# Use
explanation = "<explanation>This activation responds to references to the Pacific Ocean and geographic regions.</explanation>"
reconstructed = reconstruct(explanation)
# Compare to the original activation:
# cosine_similarity = np.dot(original, reconstructed) / (np.linalg.norm(original) * np.linalg.norm(reconstructed))
```

Working end-to-end round-trip example with the matched AV: `examples/round_trip_example.py` in the [bundled public repo](https://github.com/SolshineCode/nla-gemma-4-e2b).

## Training setup

- **Base model**: `google/gemma-4-E2B`
- **Truncation**: K+1 = 18 layers used in AR forward; hidden state captured at layer index 17 (0-indexed)
- **Quantization**: NF4 4-bit base + fp16 LoRA
- **LoRA config**: r=64, α=128, same target_modules regex as the AV
- **Linear head**: 1536 → 1536 with bias, trained in fp32, projects from L17 hidden state to reconstructed activation
- **Optimizer**: AdamW 8-bit, lr=1e-4
- **Batch**: effective batch 16
- **Max length**: 512 tokens
- **SFT steps**: 15
- **Training corpus**: same v0.0.x baseline pipeline as the AV (2,548 triples)
- **Hardware**: 4 GB GTX 1650 Ti Max-Q

## Headline numbers (v0.0.1)

- **Round-trip cosine** with the matched [v0.0.1 AV](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1): **0.438 ± 0.054** on n=42 held-out activations, 100% above the 0.30 noise floor.

## What makes this release distinctive

- **First non-Anthropic-team open-source NLA AR** at any model scale.
- **First LoRA-based NLA AR.** Anthropic's published NLA ARs are full fine-tunes at bf16. This release demonstrates a **LoRA adapter (r=64, α=128) + 1536→1536 linear head + AR truncation at K=18 layers** over NF4-quantized Gemma-4-E2B. Shipping as LoRA + small head means the AR loads in ~0.6 GB VRAM on top of the frozen NF4 base — the entire matched (AV, AR) pair fits in 4 GB. The structural-projection properties documented below are characteristic of this LoRA-AR class at 4 GB scale; they may differ at higher AR capacity / full-FT.
- **Consumer-GPU trainable.** Fits on 4 GB laptop GPU end-to-end alongside the matched AV.
- **Documented structural-projection behavior.** Standard NLA AR architectures, including this one, produce reconstructions with a strong structural-projection component independent of the input explanation. Quantitative characterization in the source research repo.

## Limitations

**NLAs can produce unexpected or incorrect explanations, and AR reconstructions can be structurally projected.** Specifically for this AR:

- **Round-trip cosine is ~97% structural-projection on this trained AR.** Replicating the published §"Measuring steganography" and §"Characterizing confabulations" tests: paraphrasing the input AV explanation moves the AR's reconstructed cosine by ~3% (Δcos paraphrase = +0.014); removing entire claims from the AV explanation moves cosine by ~0% per claim (Δcos = +0.001 per claim ablated). The reconstructed vector is approximately invariant to the explanation's content — the AR is largely projecting toward "somewhere in the L23 activation distribution" rather than reading the explanation.
- This is a **methodologically interesting finding about FVE on under-trained AR architectures**, not a unique pathology of this release. The same disaggregation should be measured on any NLA AR before relying on round-trip cosine as a content-fidelity proxy.
- **Use this AR for**: matched round-trip eval with the v0.0.1 AV (the cosine number is a valid characterization of the AV+AR pair as a system); replication of Anthropic's NLA validation pipeline at small scale; benchmarking AR-side improvements.
- **Do not use this AR for**: inferring that the AV's explanation faithfully describes the activation. Use AV-side direct content-fidelity judging instead, or in addition.

Full development history and methodology retraction notes: [`HISTORY.md`](https://github.com/SolshineCode/nla-gemma-4-e2b/blob/main/HISTORY.md). Internal experiment numbering and audit trail: source research repo (available on request).

## Citation

```bibtex
@article{frasertaliente2026nla,
  title={Natural Language Autoencoders},
  author={Fraser-Taliente, Kit and Kantamneni, Kshitij and Ong, Antonia and others},
  journal={Transformer Circuits},
  year={2026},
  url={https://transformer-circuits.pub/2026/nla/}
}

@misc{deleeuw2026nlagemma4e2bar,
  title={Gemma-4-E2B NLA AR (v0.0.1): a 4 GB consumer-GPU Activation Reconstructor},
  author={DeLeeuw, Caleb (SolshineCode)},
  year={2026},
  url={https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1}
}
```

## License

CC-BY 4.0. See [`LICENSE`](LICENSE) in the bundled repo.
