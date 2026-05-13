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

# Gemma-4-E2B Natural Language Autoencoder (Actor), v0.0.1

> First open-source NLA released independently of Anthropic's NLA team. Trained end-to-end on a 4 GB consumer GPU. The methodology contribution at small-model scale.

This is the **AV (Actor)** half of an NLA pair following the methodology of [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders) (Fraser-Taliente et al., 2026, the official Anthropic reference release). It converts residual-stream activations from `google/gemma-4-E2B` at layer 23 into natural-language explanations. As of 2026-05-12, no other open-source NLA exists outside the Fraser-Taliente reference checkpoints — verified by searching HF Hub, GitHub, and LessWrong.

For the matched AR (Critic) half, see [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](#). For the consolidated companion dataset, see [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions).

## What's distinctive about this release

- **First open-source NLA released independently of Anthropic's NLA team.** The methodology was Anthropic's (Fraser-Taliente et al. 2026). This is the first community/third-party reproduction.
- **First consumer-GPU-trainable NLA pair.** End-to-end on a 4 GB GTX 1650 Ti Max-Q laptop.
- **Honest small-model framing.** Round-trip cos = 0.438 ± 0.054 on n=42 held-out activations. 100% above the 0.30 noise-floor threshold.
- **Full reproducibility chain.** Corpus to labels to SFT to eval is single-command runnable. Every prompt versioned with SHA-256.
- **Companion dataset published independently.** 910-row Gemma-4-E2B deception/behavior completions corpus with full methodology.
- **Honest-accuracy verdict tooling.** Regression-based "descending vs flat" thresholds plus a 6-panel training dashboard.

This is the **methodology-validation small-model variant**. It's NOT a numbers-parity release with Anthropic's published 7B variants (Qwen-7B, Gemma-3-12B/27B, Llama-3.3-70B). Their cos is in the 0.7+ range. Ours is 0.438. The contribution is the **democratization path**. NLA-style interpretability becomes accessible to researchers without cluster compute.

## Honest performance summary

![Round-trip cos distribution](figures/02_round_trip_cos_distribution_v0_0_1.png)

| Metric | Value |
|---|---|
| Round-trip cosine similarity (mean) | **0.438** ± 0.054 |
| Round-trip cosine similarity (median) | 0.434 |
| Round-trip MSE (mean) | 1.124 (vs random baseline 2.0) |
| Rows above 0.30 noise-floor | **42 / 42 (100%)** |
| n_evaluated | 42 of 50 attempted. 8 produced empty AV outputs and were excluded |
| Min row cos | 0.313 |
| Max row cos | 0.558 |

**Honest failure-rate disclosure.** 16% of attempted eval rows (8 of 50) produced empty AV outputs and were excluded from the cos calculation. That is a real failure mode of the small-model variant, not a quirk of the eval set. The v0.1.x release with the diversified 9-source-family corpus and a longer SFT step budget is the test of whether scale fixes it.

See the `eval_provenance` field in the model's `nla_meta.yaml` sidecar for full reproducibility: results-JSON path, parquet SHA-256, commit SHA, paired-with-checkpoint reference, and inline headline numbers (per the eval-provenance convention introduced in our research repo).

## Architecture and training

- **Base model**. `google/gemma-4-E2B` (~2B effective params, 35 text layers)
- **Layer**. L23 (~2/3 through the text-layer stack)
- **Quantization**. NF4 4-bit base weights plus bf16 LoRA adapters
- **LoRA**. r=64, alpha=128, regex target restricting to language-model text layers (excludes audio tower):
  ```
  r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
  ```
- **Injection convention**. Forward-hook on embedding layer (Gemma-4 OOMs on `inputs_embeds`)
- **Injection scale**. `sqrt(d_model)` (Anthropic convention)
- **Context length**. 512 tokens
- **Training corpus**. 2,548 (text, activation) rows sampled from OpenWebText
- **Labeling**. gpt-4o-mini via OpenAI API (cost ~$0.50)
- **SFT steps**. 15 base plus 40 continuation, 55 total
- **Optimizer**. AdamW 8-bit, lr=1e-4 base then 3e-5 continuation
- **Batch size**. 1 sample times grad-accum 16 (effective 16)
- **Hardware**. 4 GB GTX 1650 Ti Max-Q laptop
- **Training wall time**. ~3 GPU-hours total

## Training trajectory (honest)

![Training loss](figures/01_v0_0_1_training_loss.png)

The continuation experiment shows the **SFT-saturation diagnostic**. AV training-loss flat at ~1.30 across +40 steps. AV's empty-output rate jumped from 16% to 65% in that window. This is the limit of useful single-epoch SFT on a 2,548-row corpus, not a model-capacity ceiling. The result motivates the **v0.1.0 scaling work** (in progress, corpus growing to ~11K rows across 10 diversified source families).

**Honest-accuracy verdict on the +40 continuation** (linear-regression on raw loss):

- AV. Slope = −0.0002/step, R² = 0.000. **Flat** (saturation, not descent).
- AR. Slope = −0.0015/step, R² = 0.061. **Flat** too. High variance, endpoint Δ is mostly noise.

The verdict thresholds we use (per the convention added to `CLAUDE.md`) call a training loss "descending" only when slope < −0.002 AND R² ≥ 0.10. This convention is itself a community contribution. We publish a 6-panel diagnostic dashboard tool alongside the model. See [`make_training_dashboard.py`](https://github.com/SolshineCode/deception-nanochat-sae-research/blob/main/experiments/v8_nla_local/make_training_dashboard.py).

It caught a real over-claim in our own prior session notes. The "AR was descending" interpretation of session-8's continuation was honestly flat under these thresholds.

## Companion dataset on this release

![Corpus source breakdown](figures/03_corpus_source_breakdown.png)

The 910-row consolidated Gemma-4-E2B deception/behavior completions corpus is published as a standalone artifact at [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions). It includes 70 financial-deception completions with Claude-Haiku-4-5 verdict labels, plus 840 social-role game scenarios across base and instruct variants at three layers (L10, L17, L25). Useful as Stage-0 input for any downstream small-model NLA, SAE, or interpretability work.

## Intended use

- Interpretability research on Gemma-4-E2B without requiring cluster compute
- Stage-0 input for sparse-autoencoder feature discovery on the same base model
- NLA-methodology validation in the consumer-GPU regime
- Baseline checkpoint for future small-model NLA work (e.g., Gemma-3-4B, Phi-3, Mistral-7B variants in development)
- Activation-explanation tool in deception-research or safety-monitoring pipelines (paired with a downstream classifier)

## Limitations and honest framing

- **Cos = 0.438 is far below Anthropic's published 7B numbers (~0.7+).** Use this artifact for methodology replication, not for matching their absolute performance.
- **Pair with the matched v0.0.1 AR** for round-trip eval. Mixing this AV with a third-party AR has not been validated.
- **Training corpus is OpenWebText-only.** We have an in-progress v0.1.0 release with diversified labels across 10 source families (FineWeb-Edu, Wikipedia, arXiv, in-repo Gemma-4-E2B deception completions, PKU-SafeRLHF, Anthropic/discrim-eval, Anthropic/persuasion, CAI harmless, Anthropic/llm_global_opinions) but v0.1.0 is not yet ready for HF release. Corpus is only 52% labeled at this writing.
- **No GRPO RL Phase 4** (Anthropic's v0.2.0-equivalent). On 4 GB hardware this Phase may not fit. It's under investigation.
- **Single-seed training.** Future versions will report multi-seed eval-variance.

## How to use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B", quantization_config=bnb,
                                              device_map={"": torch.cuda.current_device()})
av = PeftModel.from_pretrained(base, "Solshine/gemma-4-e2b-nla-L23-av-v0_0_1")
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
# ... see inference example in the source repo's experiments/v8_nla_local/eval_round_trip.py
```

## Citation

```bibtex
@misc{gemma4_e2b_nla_v0_0_1,
  title  = {Gemma-4-E2B Natural Language Autoencoder (v0.0.1): the first consumer-GPU-trainable open NLA},
  author = {SolshineCode},
  year   = {2026},
  month  = {may},
  url    = {https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1}
}
```

Please also cite the upstream NLA methodology when using this artifact:

- Fraser-Taliente, K., et al. (2026). *Natural Language Autoencoders*. https://transformer-circuits.pub/2026/nla/
- [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders). Anthropic's open-source NLA training pipeline.

## Acknowledgments

This release follows Anthropic's NLA methodology exactly, descoped for consumer-GPU compute. Thanks to Kit Fraser-Taliente and the Anthropic interpretability team for publishing the kitft repository as open source. Any errors in the descope are mine.

## See also

- Source research repo (full reproducibility chain). https://github.com/SolshineCode/deception-nanochat-sae-research
- Realistic-path roadmap (v0.1.0 to v0.2.0 to v1.0). `notes/V8_ANTHROPIC_GRADE_PATH_2026-05-11.md` in the source repo
- Honest-accuracy training-trend convention. See `CLAUDE.md` Research Interpretation Guardrails section
- Companion dataset. [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions)
- v0.1.0 in-progress release. Diversified-source corpus plus multi-labeler dataset coming soon (target 2-4 weeks)
- Second-model variant in development. Mistral-7B or OLMo-7B NLA, trained on rented A100 cloud GPU. Target distinct from both our local Gemma-4-E2B and Anthropic's published Qwen/Gemma-3/Llama variants

## Provenance

- **Adapter SHA-256.** See `nla_meta.yaml` sidecar.
- **Training data SHA-256.** See `nla_meta.yaml` sidecar.
- **Source repo commit SHA at training.** See `eval_provenance.commit_sha_at_training` in sidecar.
- **Eval-provenance block.** Each checkpoint's sidecar carries inline headline numbers plus paths to result JSONs plus SHA-256 of those JSONs (the eval-provenance convention from our research repo).
