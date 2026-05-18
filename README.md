# NLA-Gemma-4-E2B

**The first open-source Natural Language Autoencoder (NLA) released independently of Anthropic's NLA team — and the first NLA trained with LoRA + 4-bit quantization on a consumer GPU.** Open weights, open data, open methodology. Trained end-to-end on a 4 GB GTX 1650 Ti Max-Q laptop in ~3 GPU-hours per pair. Round-trip cosine **0.44–0.46** on held-out activations.

**Calibrated against Anthropic's deployed NLAs via the [Neuronpedia API](https://docs.neuronpedia.org/api).** Our v0.1 pair produces fluent multi-paragraph descriptive outputs in the same format class as Anthropic's deployed NLAs on Gemma-3-27B and Llama-3.3-70B (theme-correct, detail-confabulated genre). On a 10-row head-to-head, **Anthropic's NLA reads content more accurately than ours** — naming specific people (Hillary Clinton, Obama), events (2016 election), and topics where ours produces more generic linguistic-feature descriptions. The hardware gap is real: their NLAs are full-fine-tuned bf16 with GRPO post-SFT on 27B–70B base models; ours is LoRA + NF4 + SFT-only on 2B. The release is positioned as a **consumer-GPU methodology demonstration and second-source replication of the NLA training pipeline** — not as a content-fidelity peer of the Anthropic-deployed NLAs.

<img width="2752" height="1536" alt="NLA-Gemma-4-E2B release image" src="https://github.com/user-attachments/assets/7c2dbb86-03a2-4fb5-b234-b7645175825a" />

## What this is

A working pair of NLA adapters for `google/gemma-4-E2B` at residual-stream layer 23, following the methodology of Fraser-Taliente, Kantamneni, Ong et al. 2026 ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/)).

- **Activation Verbalizer (AV)**: takes a 1536-d residual-stream activation, produces a natural-language explanation.
- **Activation Reconstructor (AR)**: takes an explanation, reconstructs an activation vector that should round-trip to the original (cosine similarity).
- **Training data**: 4,734 (text, activation, persona+audit explanation) triples, labeled via a Dr. Chen / Dr. Otsuka persona+audit pipeline.

## What makes this release unique

- **First non-Anthropic-team open-source NLA at any scale.** As of 2026-05, every other NLA on HuggingFace Hub is under the `kitft` account (Kit Fraser-Taliente, the paper's first author). This is the second-source replication.
- **First LoRA + 4-bit-quantized NLA.** Anthropic's published NLAs use full fine-tuning at bf16 on H100 clusters. This work demonstrates that **LoRA adapters (r=64–80, α=128) on NF4-quantized Gemma-4-E2B** can produce NLA pairs in the same realistic output class (theme-correct, detail-confabulated) at 13× smaller parameter scale. Both halves (AV and AR) ship as LoRA adapters over a shared frozen base, so the entire pair loads into 4 GB VRAM.
- **Consumer-GPU trainable end-to-end.** NVIDIA GTX 1650 Ti Max-Q (4 GB VRAM) laptop. About 3 GPU-hours per pair. Full pipeline (Stage 0–3 + SFT + eval) on this hardware.
- **Reproducible.** Stage 0 (activation extraction) → Stage 1 (data split) → Stage 2 (LLM-judge labeling) → Stage 3 (training-format build) → SFT → round-trip eval — every step open, scripted, single-command runnable.
- **Methodology descope documented per parameter.** Conversion from H100-cluster + bf16 + full fine-tune to 4 GB + NF4 + LoRA with rationale for each choice.
- **Honest-accuracy training-trend convention.** Regression-based descending-vs-flat thresholds (raw-loss slope ≤ −0.002/step AND R² ≥ 0.10) used throughout.

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
| **AV v0.0.1** | [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1) | LoRA r=64, α=128 on Gemma-4-E2B; round-trip cosine 0.438 ± 0.054 |
| **AR v0.0.1** | [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1) | LoRA + 1536→1536 linear head, paired with AV v0.0.1 |
| **AV v0.1** (300-step, persona+audit corpus) | [`Solshine/gemma-4-e2b-nla-L23-av-v0_1_dd-step_250`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_1_dd-step_250) | LoRA r=80 + RMSNorm unfreeze + bf16; round-trip cosine 0.460 with matched AR v0.1 |
| **AR v0.1** (paraphrase-invariance retrain) | [`Solshine/gemma-4-e2b-nla-L23-ar-v0_1-paraphrase-invariant`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_1-paraphrase-invariant) | LoRA continuation from v0.0.1 AR with auxiliary paraphrase-invariance loss |
| **v0.1.x trajectory** | [`Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory) | Intermediate AV checkpoints across the v0.1.x exploration |
| **Persona+audit labeled corpus** | [`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit) | 4,734 rows, full provenance |
| **Smoke-eval dataset** | [`Solshine/gemma-4-e2b-nla-eval-smoke`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-eval-smoke) | 10 rows for `examples/round_trip_example.py` |

## Headline numbers

- **v0.0.1 round-trip cosine** (n=42 held-out activations): **0.438 ± 0.054**, 100% above the 0.30 noise floor.
- **v0.1 NLA pair round-trip cosine** (v0.1.dd AV step_250 + AR v0.1 paraphrase-invariance, n=10 held-out rl-parquet rows): **AV_OUT mean cos 0.460**.
- **Anthropic's deployed NLA round-trip cosine** for reference (Neuronpedia API, Gemma-3-27B Layer 41): **~0.99** per their API's returned `cosine_similarity` field. Their cosine is substantially higher than ours — reflects the full-FT + GRPO + 27B-base recipe vs our LoRA + NF4 + 2B-base recipe.
- **Output format class match**: multi-paragraph descriptive text in the same genre as Anthropic's NLAs. Both ship with the canonical "NLAs can produce unexpected or incorrect explanations" disclaimer.
- **Content fidelity gap from a 10-row Neuronpedia head-to-head**: Anthropic's NLA correctly names specific people / events / topics in the source text where ours produces more generic linguistic-feature descriptions. See [`results/neuronpedia_comparison_v0_1_dd_vs_gemma_27b.json`](experiments/v8_nla_local/results/content_aware_eval/neuronpedia_comparison_v0_1_dd_vs_gemma_27b.json) (in the source repo) for the per-row data + LLM-judge scores.

For the full per-checkpoint headline table see [`MODEL_CARD_AV.md`](MODEL_CARD_AV.md) and [`MODEL_CARD_AR.md`](MODEL_CARD_AR.md). For the head-to-head Neuronpedia cross-NLA calibration data + LLM-judge verdicts behind the framing above, see [`RELEASE_CALIBRATION.md`](RELEASE_CALIBRATION.md). Internal methodology investigation, experiment numbering, and audit trail are in the source research repo.

<a name="limitations"></a>
## Limitations

This release adopts the canonical NLA limitation framing — the same framing used by Anthropic's published NLAs on Neuronpedia: **NLAs can produce unexpected or incorrect explanations.** Specifically, for this release:

- **Fluent multi-paragraph descriptive output, with lower per-row content fidelity than Anthropic's deployed NLAs.** The AV produces well-formed paragraph-length descriptions in the same FORMAT class as Anthropic's published NLAs. On a 10-row Neuronpedia head-to-head against Anthropic's Gemma-3-27B Layer 41 NLA, Anthropic's NLA more accurately names the specific people, events, and topics in the source text (e.g. "Hillary Clinton's primary momentum", "Obama and Obamacare's failures") where ours produces more generic linguistic-feature descriptions ("country-specific statistical weights", "non-binary identity"). The format class match is real; the per-row content-fidelity match is not. We treat ours as a methodology demonstration at consumer-GPU scale, not a content-fidelity peer.
- **Round-trip cosine has both a content component and a structural-projection component.** Replicating Anthropic's published §"Measuring steganography" and §"Characterizing confabulations" tests shows that paraphrasing the AV output moves round-trip cosine by ~3%, and per-claim ablation moves cosine by ~0% per claim — a pattern consistent with the AR's reconstruction being dominated by a learned structural projection rather than per-token content reading. This is a property of small-model + LoRA NLAs in general, not a unique pathology of this release. Use AV-side per-row content-fidelity judging (validity × specificity × relatedness rubric) alongside round-trip cosine.
- **Hardware-bound quality ceiling.** Numbers reflect a single 4 GB GTX 1650 Ti Max-Q with NF4 + LoRA + ~5K-row corpus + ≤300 SFT steps. Larger GPUs with bf16 + full fine-tune + larger corpus + GRPO post-SFT (the recipe Anthropic uses) would likely raise quality further.
- **Use this release for**: consumer-GPU NLA research, methodology benchmarking, replication of Anthropic's NLA validation pipeline at small scale, per-feature interpretability exploration with the canonical NLA caveat.
- **Do not use this release for**: drawing strong claims about a specific activation from a single AV output without independent verification (the same constraint that applies to all currently-published NLAs).

Full development history including methodology retraction and process notes: see [`HISTORY.md`](HISTORY.md). Internal experiment numbering, audit trail, and supplementary methodology investigation: in the source research repo (`SolshineCode/deception-nanochat-sae-research`, available on request).

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
