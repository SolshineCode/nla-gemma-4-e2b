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
  - sparse-autoencoder
  - gemma
  - consumer-gpu
  - peft
  - lora
library_name: peft
pipeline_tag: text-generation
---

# Gemma-4-E2B Natural Language Autoencoder (Critic), v0.0.1

> First open-source NLA released independently of Anthropic's NLA team. Trained end-to-end on a 4 GB consumer GPU. The methodology contribution at small-model scale.

This is the **AR (Critic)** half of an NLA pair following the methodology of [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders) (Fraser-Taliente et al., 2026, the official Anthropic reference release). It takes a natural-language explanation and reconstructs the residual-stream activation it was meant to describe — the round-trip test for AV-explanation faithfulness.

For the matched AV (Actor) half, see [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1). For the consolidated companion dataset, see [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions).

## Architecture and training

- **Base model**. `google/gemma-4-E2B` truncated to the first 18 of 35 text layers + Linear(1536, 1536) head
- **Layer**. L23 (~2/3 through the text-layer stack; reconstruction target layer)
- **Quantization**. NF4 4-bit base weights plus bf16 LoRA adapters
- **LoRA**. r=64, alpha=128, regex target restricting to language-model text layers (excludes audio tower):
  ```
  r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
  ```
- **AR prompt template**. `Summary of the following text: <text>{explanation}</text> <summary>`
- **Suffix-anchored extraction**. AR extracts the reconstructed activation from the FINAL token (the trailing space after `<summary>`)
- **MSE objective**. L2-normalized prediction vs L2-normalized target, both scaled to `sqrt(d_model)`
- **Optimizer**. AdamW 8-bit, lr=1e-4 base then 3e-5 continuation
- **Training wall time**. ~1.5 GPU-hours

This is the **first half-precision LoRA AR** of the public NLA family — Anthropic's full-finetune AR variants need 14+ GB; this fits on 4 GB.

## Honest performance summary

| Metric | Value |
|---|---|
| Round-trip cosine similarity (mean) | **0.438** ± 0.054 |
| Round-trip cosine similarity (median) | 0.434 |
| Round-trip MSE (mean) | 1.124 (vs random baseline 2.0) |
| Rows above 0.30 noise-floor | **42 / 42 (100%)** |
| n_evaluated | 42 of 50 attempted. 8 produced empty AV outputs and were excluded |
| Min row cos | 0.313 |
| Max row cos | 0.558 |

Round-trip cos is the matched AV+AR pair on held-out OpenWebText activations. The eval is symmetric: round-trip cos quality is co-determined by both halves of the pair, so this number is the AR's quality marker as much as the AV's.

## Intended use

- Round-trip eval of any candidate AV checkpoint targeting Gemma-4-E2B L23
- Stage-0 input for cross-AR ablation studies in the consumer-GPU regime
- Baseline AR for future small-model NLA work targeting Gemma-4-E2B
- Activation-reconstruction component in deception-research or safety-monitoring pipelines

## Limitations

- Pair with the matched v0.0.1 AV. Mixing this AR with a third-party AV has not been validated.
- Training corpus is OpenWebText-only. v0.1.0 with diversified labels across 10 source families is in progress.
- Round-trip cos = 0.438 is below Anthropic's published 7B numbers (~0.7+). Use for methodology replication, not absolute performance matching.
- Linear(1536, 1536) head is loaded separately from the LoRA adapter (`linear_head.pt`).

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B", quantization_config=bnb,
                                             device_map={"": torch.cuda.current_device()})
ar = PeftModel.from_pretrained(base, "Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1")
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
linear_head_state = torch.load("linear_head.pt")  # download separately from this repo
# ... see inference example in the source repo's experiments/v8_nla_local/eval_round_trip.py
```

## Citation

```bibtex
@misc{gemma4_e2b_nla_v0_0_1_ar,
  title  = {Gemma-4-E2B Natural Language Autoencoder (Critic) v0.0.1: the first consumer-GPU-trainable open NLA},
  author = {SolshineCode},
  year   = {2026},
  month  = {may},
  url    = {https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1}
}
```

Please also cite the upstream NLA methodology:

- Fraser-Taliente, K., et al. (2026). *Natural Language Autoencoders*. https://transformer-circuits.pub/2026/nla/
- [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders). Anthropic's official reference NLA training pipeline.

## See also

- Source research repo. https://github.com/SolshineCode/deception-nanochat-sae-research
- Matched AV. [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1)
- Companion dataset. [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions)
