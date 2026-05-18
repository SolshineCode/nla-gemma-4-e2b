# NLA-Gemma-4-E2B

**The first open-source Natural Language Autoencoder (NLA) released independently of Anthropic's NLA team.** Trained end-to-end on a single 4 GB consumer GPU. Open weights, open data, open methodology. *(NLA weighs herein. Work in progress — actively iterating on training methodology and content-fidelity evaluation.)*

<img width="2752" height="1536" alt="NLA-Gemma-4-E2B release image" src="https://github.com/user-attachments/assets/7c2dbb86-03a2-4fb5-b234-b7645175825a" />

## What this is

A working pair of NLA adapters for `google/gemma-4-E2B` at residual-stream layer 23, following the methodology of Fraser-Taliente, Kantamneni, Ong et al. 2026 ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/)).

- **Activation Verbalizer (AV)**: takes a 1536-d residual-stream activation, produces a natural-language explanation.
- **Activation Reconstructor (AR)**: takes an explanation, reconstructs an activation vector that should round-trip to the original (cosine similarity).
- **Training data**: 4,734 (text, activation, persona+audit explanation) triples, labeled via a Dr. Chen / Dr. Otsuka persona+audit pipeline.

## What makes this release unique

- **First non-Anthropic-team open-source NLA at any scale.** As of 2026-05, every other NLA on HuggingFace Hub is under the `kitft` account (Kit Fraser-Taliente, the paper's first author). This is the second-source replication.
- **First LoRA-based NLA training.** Anthropic's published NLAs use full fine-tuning at bf16 on H100 clusters. This work demonstrates that **LoRA adapters (r=64–80, α=128) on NF4-quantized Gemma-4-E2B** can produce NLA pairs at the same realistic output class (theme-correct, detail-confabulated) at 13× smaller parameter scale. The LoRA + 4-bit-quant + RMSNorm-unfreeze stack is the architectural choice that makes this entire 4 GB-feasible methodology possible — both halves of the pair (AV and AR) ship as LoRA adapters over the same frozen base, so loading the full pair into 4 GB VRAM is feasible.
- **Consumer-GPU trainable.** End-to-end on an NVIDIA GTX 1650 Ti Max-Q (4 GB VRAM) laptop. About 3 GPU-hours for the v0.0.1 pair. Full pipeline (Stage 0–3 + SFT + eval) runs on this hardware because of the LoRA + NF4 stack above.
- **Reproducible.** Stage 0 (activation extraction) → Stage 1 (data split) → Stage 2 (LLM-judge labeling) → Stage 3 (training-format build) → SFT → round-trip eval — every step open, scripted, single-command runnable.
- **Methodology descope documented.** The conversion from Anthropic's H100-cluster + bf16 + full fine-tune to 4 GB + NF4 + LoRA is documented per parameter, with rationale. Both the gains (4 GB feasibility, faster iteration) and the limitations (the L2 per-row identity bottleneck documented in `ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md` Addendum 4) are surfaced honestly.
- **Honest-accuracy training-trend convention.** Regression-based descending-vs-flat thresholds (raw-loss slope ≤ −0.002/step AND R² ≥ 0.10) caught a false-positive trend during development. Default in this repo.

## Quick start

```bash
git clone https://github.com/SolshineCode/nla-gemma-4-e2b
cd nla-gemma-4-e2b
pip install -r requirements.txt
python examples/round_trip_example.py
```

The example loads the published v0.0.1 AV + AR adapters from HuggingFace, samples 10 activations from the bundled smoke-eval dataset, generates explanations, reconstructs activations, and prints round-trip cosine similarities.

## What's in this release

| Artifact | Location | Notes |
|---|---|---|
| **AV v0.0.1** | [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1) | LoRA r=64, α=128 on Gemma-4-E2B; in-distribution `injection_scale = sqrt(d_model) ≈ 39` |
| **AR v0.0.1** | [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1) | LoRA + 1536→1536 linear head on Gemma-4-E2B L17 |
| **v0.1.x trajectory** | [`Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory) | Intermediate AV checkpoints across the v0.1.x exploration; trajectory README documents which configs were valid vs out-of-distribution |
| **Persona+audit labeled corpus** | [`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit) | 4,734 rows, full provenance |
| **Smoke-eval dataset** | [`Solshine/gemma-4-e2b-nla-eval-smoke`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-eval-smoke) | 10 rows for `examples/round_trip_example.py` |

## Headline numbers (v0.0.1, the recommended pair)

- **Round-trip cosine** (n=42 held-out activations): **0.438 ± 0.054**, 100% above the 0.30 noise floor.
- **AV under SFT slope** (linear regression on raw loss across SFT steps): consistent with descending-and-converged at step 15.
- **Per-row output diversity** (unique 60-char prefixes / 10 rows): present but limited; ~80% template-like at v0.0.1, with content-conditional fill-in slots.

For the full per-checkpoint headline table see [`MODEL_CARD_AV.md`](MODEL_CARD_AV.md) and [`MODEL_CARD_AR.md`](MODEL_CARD_AR.md).

<a name="limitations"></a>
## Limitations

This release adopts the canonical NLA limitation framing: **NLAs can produce unexpected or incorrect explanations.** Specifically, for this release:

- **Thematic-correctness with detail-confabulation is the realistic output class.** The AV typically identifies the broad topic of the activation correctly (genre of source text, dominant entity type, structural pattern) and confabulates specific tokens, names, or examples that don't appear in the source. This matches the documented qualitative behavior of larger NLAs in the published literature; the small-model version shows more confabulation per output than the reference.
- **Round-trip cosine has a structural-projection component.** Replicating the published §"Measuring steganography" and §"Characterizing confabulations" tests on v0.0.1 shows that paraphrasing the AV output moves the round-trip cosine by ~3%, and removing entire claims from the AV output moves cosine by ~0% per claim. Most of the v0.0.1 round-trip-cosine signal is the AR's structural projection toward "somewhere in OpenWebText L23 activation space," not the explanation's specific content. Use AV-side per-row content-fidelity judging (validity × specificity × relatedness rubric) alongside round-trip cosine, never round-trip cosine alone. This is a methodologically interesting finding about FVE on under-trained AR architectures.
- **Hardware-bound quality ceiling.** All numbers in this release reflect a single 4 GB GTX 1650 Ti Max-Q with NF4 quantization + LoRA + small (<5K row) corpus + ≤300 SFT steps. Reproducing on a larger GPU with bf16 + full fine-tune + larger corpus would close some of the qualitative gap with the published reference NLAs.
- **Not a production interpretability tool.** Use for methodology benchmarking, infrastructure replication, and consumer-GPU NLA research. Do not draw strong claims about a specific activation's content from a single AV explanation.

The full development history including a methodology-bug retraction and an autonomous-research-process retrospective is in [`HISTORY.md`](HISTORY.md). It documents how an autonomous Claude research assistant introduced an uncited code-comment claim that propagated through 5 training runs over 8 days before being caught, and the 5 specific process changes added to prevent recurrence.

## Reproducing the training

```bash
# 1. Activation extraction (Stage 0)
python stage0_data_gen.py --output data/stage1/

# 2. Stage 1 split (60/20/20 doc-level)
python stage1_split.py --input data/stage1/

# 3. Label with persona+audit pipeline (Gemini CLI; free under subscription)
python stage2_gemini_explain.py --persona expert --audit --limit 4734

# 4. Build training format (Stage 3)
python stage3_build.py --output data/stage3/

# 5. Train AV (LoRA, NF4, ~2h on 4 GB GTX 1650 Ti)
python stage_av_sft.py \
    --train-data data/stage3/av_sft.parquet \
    --output checkpoints/av_v0/ \
    --max-steps 15

# 6. Train AR
python stage_ar_sft.py \
    --train-data data/stage3/ar_sft.parquet \
    --output checkpoints/ar_v0/ \
    --max-steps 15

# 7. Round-trip eval
python round_trip_eval.py \
    --av checkpoints/av_v0/final \
    --ar checkpoints/ar_v0/final \
    --eval-data data/stage1/rl.parquet \
    --n-rows 50
```

For full source (`stage0_data_gen.py`, `stage1_split.py`, etc.) see the research repo `SolshineCode/deception-nanochat-sae-research` (available on request).

## Hardware

- **Training**: NVIDIA GTX 1650 Ti Max-Q, 4 GB VRAM (laptop). NF4 4-bit base + bf16/fp16 LoRA adapters. ~3 GPU-hours for v0.0.1 end-to-end.
- **Inference**: any GPU that fits Gemma-4-E2B in NF4 (~2 GB) or full bf16 (~6 GB).

## Citation

If you use this release, please cite both the underlying methodology and this artifact:

```bibtex
@article{frasertaliente2026nla,
  title={Natural Language Autoencoders},
  author={Fraser-Taliente, Kit and Kantamneni, Kshitij and Ong, Antonia and others},
  journal={Transformer Circuits},
  year={2026},
  url={https://transformer-circuits.pub/2026/nla/}
}

@misc{deleeuw2026nlagemma4e2b,
  title={NLA-Gemma-4-E2B: A 4 GB consumer-GPU Natural Language Autoencoder for Gemma-4-E2B (v0.0.1)},
  author={DeLeeuw, Caleb (SolshineCode)},
  year={2026},
  url={https://github.com/SolshineCode/nla-gemma-4-e2b}
}
```

## License

CC-BY 4.0 for the weights, datasets, and documentation. Apache 2.0 for the training and eval scripts. See `LICENSE`.

## Acknowledgments

Methodology: Kit Fraser-Taliente, Kshitij Kantamneni, Antonia Ong, and coauthors for the underlying NLA framework and the public `kitft/natural_language_autoencoders` reference repo. The methodology, prompt templates, and evaluation framework here are direct adaptations of that work. Any errors in the descope-to-consumer-hardware reduction are mine; see [`HISTORY.md`](HISTORY.md) for the documented mistakes and recoveries.
