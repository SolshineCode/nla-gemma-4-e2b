# NLA-Gemma-4-E2B

**The first open-source Natural Language Autoencoder (NLA) released independently of Anthropic's NLA team — and the first NLA trained with LoRA + 4-bit quantization on a consumer GPU.** Open weights, open data, open methodology. Trained end-to-end on a 4 GB GTX 1650 Ti Max-Q laptop in ~3 GPU-hours per pair. Round-trip cosine **0.44–0.46** on held-out activations.

**Calibrated against Anthropic's deployed NLAs via the [Neuronpedia API](https://docs.neuronpedia.org/api).** Our v0.1 pair produces fluent multi-paragraph descriptive outputs in the same format class as Anthropic's deployed NLAs on Gemma-3-27B and Llama-3.3-70B (a genre-plausible, detail-confabulated register). On a **n=50 head-to-head** (Claude judge 49/50 preferred Anthropic; Gemini judge with explicit param-size-gap calibration 48/49 preferred Anthropic), **Anthropic's deployed NLAs read content more accurately than ours** — naming specific people (Hillary Clinton, Obama), events (2016 election), and topics where ours produces more generic linguistic-feature descriptions. A direct content-specificity retrieval eval (can each AV output recover the source document it came from?) puts our v0.1 AV **at chance** across lexical, semantic, and two LLM-judge probes, so the output is format-plausible but **not per-row content- or theme-discriminative**, not merely "theme-correct." The encouraging part, though, is that the content really is in there to read: a ceiling test recovers the source document well above chance straight from the raw L23 activation (doc-level retrieval 0.24 vs 0.077 chance; a logistic probe reads 13-way document identity at **60%**), so the bottleneck is in **how the verbalizer reads the injected activation, not the 2B model's representation of the content** — the AV largely falls back on a learned prior instead of surfacing what the activation encodes. Getting it to surface that content is an open problem we're actively working on, not a settled matter of training budget. A layer sweep even points at **L17 carrying more than 2× the document signal of L23**, an obvious lever for the next version. Full suite, data, and scripts: [`experiments/v8_nla_local/CONTENT_SPECIFICITY_EVAL.md`](experiments/v8_nla_local/CONTENT_SPECIFICITY_EVAL.md). Two factors compound the gap, and we cannot disentangle them with current data: **(a) the training-stack gap** — Anthropic's NLAs are full-FT bf16 + GRPO post-SFT, ours is LoRA + NF4 + SFT-only at a few hundred SFT steps; **(b) the cross-model activation gap** — their NLAs read 27B-Gemma-3 / 70B-Llama-3.3 activations, ours reads 2B-Gemma-4-E2B activations on the same source text, and the underlying activations may intrinsically differ in per-instance specificity due to scale-dependent polysemanticity. See [`RELEASE_CALIBRATION.md`](RELEASE_CALIBRATION.md) for the full caveat + the proposed clean disentangling experiment (~30-50 A100-hr on cloud GPU). The release is positioned as a **consumer-GPU methodology demonstration and second-source replication of the NLA training pipeline** — not as a content-fidelity peer of the Anthropic-deployed NLAs. **There is no published reference NLA for Gemma-4-E2B L23 specifically; ours is the only one that reads this model+layer.**

<img width="2752" height="1536" alt="NLA-Gemma-4-E2B release image" src="https://github.com/user-attachments/assets/7c2dbb86-03a2-4fb5-b234-b7645175825a" />

**Best pair (v0.1):**
- **AV:** [`Solshine/gemma-4-e2b-nla-L23-av-v0_1_dd-step_250`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_1_dd-step_250)
- **AR:** [`Solshine/gemma-4-e2b-nla-L23-ar-v0_1-paraphrase-invariant`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_1-paraphrase-invariant)

## What this is

A working pair of NLA adapters for `google/gemma-4-E2B` at residual-stream layer 23, following the methodology of Fraser-Taliente, Kantamneni, Ong et al. 2026 ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/)).

- **Activation Verbalizer (AV)**: takes a 1536-d residual-stream activation, produces a natural-language explanation.
- **Activation Reconstructor (AR)**: takes an explanation, reconstructs an activation vector that should round-trip to the original (cosine similarity).
- **Training data**: 4,734 (text, activation, persona+audit explanation) triples, labeled via a Dr. Chen / Dr. Otsuka persona+audit pipeline.

## What makes this release unique

- **First non-Anthropic-team open-source NLA at any scale.** As of 2026-05, every other NLA on HuggingFace Hub is under the `kitft` account (Kit Fraser-Taliente, the paper's first author). This is the second-source replication.
- **First LoRA + 4-bit-quantized NLA.** Anthropic's published NLAs use full fine-tuning at bf16 on H100 clusters. This work demonstrates that **LoRA adapters (r=64–80, α=128) on NF4-quantized Gemma-4-E2B** can produce NLA pairs in the same realistic output class (genre-plausible, detail-confabulated) at 13× smaller parameter scale. Both halves (AV and AR) ship as LoRA adapters over a shared frozen base, so the entire pair loads into 4 GB VRAM.
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
- **The gap is the verbalizer's, not the activation's**: a ceiling test on the raw L23 activations recovers the source document well above chance (doc-level retrieval **0.24** vs 0.077; a logistic probe reads 13-way document identity at **60%**), and a layer sweep finds **L17 carrying more than 2× the document signal of L23**. The content the AV does not yet surface is demonstrably present in the activation, so the per-row gap is about the verbalizer's reading of the activation — an open research direction — rather than a ceiling of the 2B model. How much finer per-instance specificity the 2B activation carries is something we're still investigating.

For the full per-checkpoint headline table see [`MODEL_CARD_AV.md`](MODEL_CARD_AV.md) and [`MODEL_CARD_AR.md`](MODEL_CARD_AR.md). For the head-to-head Neuronpedia cross-NLA calibration data + LLM-judge verdicts behind the framing above, see [`RELEASE_CALIBRATION.md`](RELEASE_CALIBRATION.md). Internal methodology investigation, experiment numbering, and audit trail are in the source research repo.

## Why this SFT pair, not a GRPO checkpoint

The Anthropic NLA recipe has four phases: Stages 0–3 (data + labeling) → SFT → **Phase 4 GRPO** (joint RL fine-tune of the AV with the AR's reconstruction-MSE as reward). The v0.0.1 and v0.1 pairs published here are the **SFT-only** output (Phases 1–3); Phase 4 GRPO was deferred at first release because it had not yet been adapted to the 4 GB hardware regime.

Between **2026-05-25 and 2026-05-29** the deferred Phase 4 was implemented and run end-to-end on the same 4 GB GTX 1650 Ti Max-Q, with alternating AV/AR loads and R=4 rollout batching to fit in VRAM. The trial swept **5 reward formulations × 4 entropy regimes across 120 rollouts**, with intermediate L2 cross-row-argmax readouts at rollouts 40, 60, 80, 100, 120. The outcome:

| Rollout | Reward | Entropy β | L2 cross-row argmax (n=10) | AV output |
|---:|---|---:|---:|---|
| 40 | MSE | 0.0 | 0.100 (chance) | coherent multi-paragraph (same class as SFT v0.1) |
| 60 | MSE | 0.3 | 0.100 (chance) | random Unicode tokens — degenerate |
| 80 | contrastive-mean | 1.0 | 0.100 (chance) | whitespace-only — degenerate |
| 100 | contrastive-max | 1.0 | 0.100 (chance) | "evasion evasion evasion …" mode collapse |
| 120 | contrastive-max + AR-contrastive | 0.1 | 0.100 (chance) | "evasion evasion evasion …" mode collapse |

**No GRPO checkpoint clears the bar of the released SFT pair.** The only GRPO checkpoint that preserved coherent AV output (r40, MSE reward + no entropy bonus) matched the SFT v0.1 L2 margin within noise — it did not beat the released pair on the headline metric, so shipping it would add nothing. Every higher-entropy checkpoint destroyed the AV's interpretability surface (gibberish, whitespace, or mode-collapsed output) without compensating with any measurable per-row-fidelity gain.

**The released SFT pair is therefore strictly better than any GRPO checkpoint we produced on this hardware**: both classes are at L2 = chance on per-row identity, but the released SFT pair preserves the coherent multi-paragraph descriptive output that gives the NLA pipeline its interpretability surface.

**What this trial contributes to the research record.** Combining the prior 8-attempt SFT lever sweep with this 5-readout GRPO sweep yields **14 distinct training attempts spanning the full Anthropic recipe**, all converging to L2 = chance at 4 GB. The L2 ceiling at this hardware scale is robust to optimizer/loss/scheduler levers within SFT, to reward shape (MSE vs contrastive vs contrastive-max), to entropy regularization, and to training paradigm (SFT-only vs SFT+GRPO). The open question — base-model scale (2B vs 27B/70B) vs hardware constraint (NF4 + LoRA + small contrast pool) as the dominant bottleneck — would be answered by a cross-model recipe-controlled retrain on Gemma-3-27B, flagged for follow-on grant-funded work. See [`RELEASE_CALIBRATION.md`](RELEASE_CALIBRATION.md) §"Addendum 2026-05-29" for the full per-checkpoint reward/entropy/output tables.

<a name="limitations"></a>
## Limitations

This release adopts the canonical NLA limitation framing — the same framing used by Anthropic's published NLAs on Neuronpedia: **NLAs can produce unexpected or incorrect explanations.** Specifically, for this release:

- **Fluent multi-paragraph descriptive output, with lower per-row content fidelity than Anthropic's deployed NLAs.** The AV produces well-formed paragraph-length descriptions in the same FORMAT class as Anthropic's published NLAs. On a 10-row Neuronpedia head-to-head against Anthropic's Gemma-3-27B Layer 41 NLA, Anthropic's NLA more accurately names the specific people, events, and topics in the source text (e.g. "Hillary Clinton's primary momentum", "Obama and Obamacare's failures") where ours produces more generic linguistic-feature descriptions ("country-specific statistical weights", "non-binary identity"). The format class match is real; the per-row content-fidelity match is not. A direct content-specificity retrieval eval (does each AV output recover its own source document among the 13 in the held-out set?) puts the v0.1 AV at chance across lexical, semantic, and two LLM-judge probes (doc-level top-1 0.08 to 0.10 vs 0.077 chance; judge accuracy 0.24 to 0.27 vs 0.20 chance; all non-significant). So the v0.1 output is diverse (45/50 unique strings) but **not per-row content- or theme-discriminative**: the diversity is decoupled from source content. The one honest scope-limit is that a feature constant across all 13 news documents (for example "formal news register") would be real content yet invisible to this eval. **Importantly, the gap is the verbalizer's, not the activation's:** a ceiling test on the raw L23 activations recovers the source document well above chance (doc-level retrieval 0.24 vs 0.077; a logistic probe reads 13-way document identity at 60%), so the content the AV fails to surface is demonstrably present in the activation. That places the bottleneck in the verbalizer's reading of the injected activation rather than in the 2B model's content — an open problem we're actively investigating, not a settled matter of training budget. (How much finer per-instance specificity the 2B activation carries is a separate, still-open question; see [`RELEASE_CALIBRATION.md`](RELEASE_CALIBRATION.md).) A layer sweep adds that L23 is not even the most content-discriminative site (L17 is more than 2x higher), so a future NLA could also retarget the layer. Full methods, numbers, and per-trial data: [`experiments/v8_nla_local/CONTENT_SPECIFICITY_EVAL.md`](experiments/v8_nla_local/CONTENT_SPECIFICITY_EVAL.md). We treat ours as a methodology demonstration at consumer-GPU scale, not a content-fidelity peer.
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
