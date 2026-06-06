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

Trained end-to-end on a single 4 GB consumer GPU (NVIDIA GTX 1650 Ti Max-Q) following a **customized variation (see below)** of the methodology of Fraser-Taliente, Kantamneni, Ong et al. 2026 ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/)).

### Customizations vs the source methodology
A consumer-hardware *variation*, not a faithful reproduction: a **LoRA adapter (+ a 1536→1536 linear head) over a 4-bit NF4-quantized** frozen base rather than full-fine-tune bf16; a single 4 GB GPU with micro-batch 1 + gradient accumulation and a modest SFT-step budget; an **SFT-only** released pair (Phase-4 GRPO explored separately, not shipped); and added AV-side evaluations beyond round-trip cosine (see the matched [`MODEL_CARD_AV.md`](MODEL_CARD_AV.md)). Note round-trip cosine on this AR is largely a structural-projection metric, not per-row faithfulness — see Limitations.

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

## Release rationale: why this SFT pair and not a GRPO checkpoint

The Anthropic NLA recipe (Fraser-Taliente et al. 2026) has four phases: Stages 0–3 (data + labeling) → SFT (supervised fine-tune of the AV+AR pair) → **Phase 4 GRPO** (joint REINFORCE-style RL fine-tune of the AV with the AR's reconstruction-MSE as reward signal, plus an AR "keep-up" SFT update and a KL anchor). The published `v0.0.1` and `v0.1` pairs are the **SFT-only** output of Phases 1–3; Phase 4 GRPO was deferred at first release because it had not yet been adapted to the 4 GB hardware regime.

Between 2026-05-25 and 2026-05-29 the deferred Phase 4 was implemented and run **end-to-end on the same 4 GB GTX 1650 Ti Max-Q**, with alternating AV/AR loads and R=4 rollout batching to fit in VRAM. The trial swept **5 reward formulations × 4 entropy regimes across 120 rollouts**. At every intermediate L2 readout (rollouts 40, 60, 80, 100, 120) the GRPO-updated AV+AR pair scored **L2 cross-row-argmax = 0.100 (chance)** on the n=10 held-out RL eval — the same as the SFT v0.1 baseline pair. Higher-entropy configurations additionally produced degenerate AV outputs (random Unicode tokens, whitespace, or "evasion evasion evasion…" mode collapse).

**Verdict for this AR.** No GRPO AR checkpoint is shipped. Within the GRPO trial the AR was updated under MSE-keep-up on the AV's rollout outputs (and, at rollout 108–120, briefly under a contrastive AR-loss variant). The post-GRPO AR's reconstruction quality on held-out activations did not improve over this v0.0.1 SFT AR — the round-trip cosine and L2 cross-row-argmax both stayed in the same noise band as the released pair. The released v0.0.1 AR (and the v0.1 paraphrase-invariance AR variant) therefore remains the recommended Activation Reconstructor for this hardware/model class.

**Research contribution.** Combining the 8-attempt SFT lever sweep with the 5-readout GRPO sweep yields **14 distinct training attempts spanning the full Anthropic recipe**, all converging to L2 = chance at 4 GB. The L2 ceiling at this hardware scale is robust to (a) optimizer-/loss-/scheduler-side levers within SFT, (b) reward shape (MSE vs contrastive vs contrastive-max), (c) entropy regularization (β ∈ {0, 0.1, 0.3, 1.0}), and (d) training paradigm (SFT-only vs SFT+GRPO). The structural-projection signature of the released AR (Δcos ≈ 0 per per-claim ablation) is now characterized as a 4 GB-LoRA-AR property robust to GRPO updates, not an artifact of incomplete training. The open question — whether the bottleneck is **base-model scale** (2B vs 27B/70B) or **the 4 GB hardware constraint** (NF4 + LoRA + small contrast pool) — would be answered by a cross-model recipe-controlled retrain on Gemma-3-27B; that experiment is flagged for follow-on grant-funded work.

The v0.0.1 + v0.1 SFT AR pair on this repo therefore represents the **best-coherent-pair checkpoint** from a comprehensive characterization of the Anthropic NLA recipe at 4 GB, **not** a checkpoint that ran out of training budget before further phases could be attempted.

## Limitations

**NLAs can produce unexpected or incorrect explanations, and AR reconstructions can be structurally projected.** Specifically for this AR:

- **Round-trip cosine is ~97% structural-projection on this trained AR.** Replicating the published §"Measuring steganography" and §"Characterizing confabulations" tests: paraphrasing the input AV explanation moves the AR's reconstructed cosine by ~3% (Δcos paraphrase = +0.014); removing entire claims from the AV explanation moves cosine by ~0% per claim (Δcos = +0.001 per claim ablated). The reconstructed vector is approximately invariant to the explanation's content — the AR is largely projecting toward "somewhere in the L23 activation distribution" rather than reading the explanation. For reference, Anthropic's deployed Gemma-3-27B AR reports round-trip cosine **~0.99** via the Neuronpedia API on the same shape of input; the gap to our 0.44–0.46 quantifies the hardware / methodology distance. This is a property of the AR's reconstruction at 4 GB, not a statement that the content is missing: an independent AV-side ceiling test recovers the source document from the raw L23 activation at **60%** linear-probe accuracy, so the system-level per-row gap is about how the verbalizer and reconstructor read the activation — an open research question — not an absence of content in the activation (see the matched AV card and the repo README).
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
