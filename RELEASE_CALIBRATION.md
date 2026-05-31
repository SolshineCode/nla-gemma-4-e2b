# Release Calibration — Cross-NLA Comparison

A head-to-head comparison between this release (Gemma-4-E2B v0.1, LoRA + 4-bit, consumer-GPU trained) and Anthropic's deployed NLAs via the [Neuronpedia API](https://docs.neuronpedia.org/api). Performed 2026-05-18 as part of release preparation.

This document is the source-of-truth honest positioning for what this release does and does not match.

## Headline numbers

**n=10 head-to-head on the same source texts**, last-token activation, against two Anthropic NLAs available via Neuronpedia API:

| Metric | Anthropic Gemma-3-27B (kitft-l41) | Anthropic Llama-3.3-70B (kitft-l53) | Our v0.1 (Gemma-4-E2B) |
|---|---:|---:|---:|
| Round-trip cosine (their API field / our eval) | ~0.99 | ~0.99 | 0.46 |
| LLM-judge validity (1-5, n=10) | **3.1** | **3.2** | **1.0** |
| LLM-judge specificity (1-5, n=10) | **3.2** | **3.3** | **1.2** |
| Judge preference (out of 10 rows) | **10** | **10** | **0** |

Across both Anthropic NLAs the picture is identical: **20/20 judge preferences for Anthropic, 0 for ours, 0 ties.** Even Anthropic's deployed NLAs average only 3.1-3.3 on a 1-5 validity scale ("partially correct theme") — they're good but not perfect; the canonical "NLAs can produce unexpected or incorrect explanations" caveat applies to all NLAs at any scale. But the gap between Anthropic's ~3.15 and our 1.0 is real and reproducible across two independent reference NLAs.

Per-row data + judge verdicts: `experiments/v8_nla_local/results/content_aware_eval/neuronpedia_*.json` in the source research repo.

## What this comparison shows

**Same output FORMAT class.** Both Anthropic's NLAs and ours produce multi-paragraph descriptive text in the same canonical "NLAs can produce unexpected or incorrect explanations" genre. Both ship with that disclaimer.

**Different per-row content fidelity.** Anthropic's NLAs name specific people, events, dates, authors:

| Source actual topic | Anthropic NLA names | Our NLA produces |
|---|---|---|
| Hillary Clinton primary news | "Hillary Clinton's primary/caucus momentum" | "country-specific statistical weights" |
| Obama op-ed | "Obama and Congressional Republicans, Obamacare's failures" / "Joseph Klein, Fox News, David Goldman" | "country-specific statistical data, high-precision numeric register" |
| 2016 election results | "2016 tally update" / "2008 New Hampshire primary" | "1918 influenza pandemic" |
| Football player news | "summary/discussion about a football player's falling" / "Boom Williams, Kentucky running back" | "the 'unnatural' or 'unnatural' semantic field" |

Our outputs are template-clustered ("country-specific statistical weights" appears on rows 0, 4, and 9 across THREE different source documents). Anthropic's outputs name specific entities consistent with the source.

## Honest positioning

**This release is**:

- The first non-Anthropic-team open-source NLA at any scale
- The first NLA trained with **LoRA + 4-bit quantization** on a consumer GPU (4 GB GTX 1650 Ti Max-Q)
- A second-source replication of the NLA training pipeline (Fraser-Taliente et al. 2026)
- A methodology demonstration showing the NLA pipeline runs end-to-end on consumer hardware

**This release is NOT**:

- A content-fidelity peer of Anthropic's deployed Gemma-3-27B or Llama-3.3-70B NLAs
- Suitable as the sole interpretability tool for drawing strong claims about a specific activation

The hardware / methodology gap explains the fidelity gap:

| | Anthropic deployed | This release |
|---|---|---|
| Base model | 27B–70B params | 2B params (13–35× smaller) |
| Quantization | bf16 | NF4 4-bit |
| Adaptation | full fine-tune | LoRA (r=64–80) |
| Post-SFT | GRPO with frontier-judge content reward | none (SFT only) |
| Training corpus | undisclosed, large | ~5K rows |
| Hardware | H100 clusters | 4 GB consumer laptop |
| GPU-hours per pair | undisclosed, large | ~3 |

## What to use this release for

- **Research on consumer-GPU NLA training**: methodology benchmarking, AR-side experiments, training-trend convention testing
- **Replication of Anthropic's NLA validation pipeline at small scale**: same Stage 0-3 + SFT + eval structure, runs in ~3 GPU-hours
- **Baseline for AR-bottleneck investigation**: the L1 (surface-form) and L2 (per-row identity) decomposition of structural projection is the cleanest characterization available, with cross-row identity as the headline metric for any future AR retrain

## What to use Anthropic's deployed NLAs for instead (via Neuronpedia API)

- Drawing content-aware interpretability claims about a specific Gemma-3-27B or Llama-3.3-70B activation
- Per-row analysis of model behavior at the activation level
- Any downstream task that requires per-instance content reading rather than format-class output

## Methodology of this comparison

1. **Sample**: 10 rows initially, extended to n=50 on 2026-05-19, from a held-out RL eval parquet (politically-themed news / op-eds; Fineweb-style source texts)
2. **Activation site**: last token of each source text after running through the target model
3. **Anthropic side**: `POST /api/nla/explain` on neuronpedia.org with each (text, last_position) pair — running against Anthropic's deployed Gemma-3-27B or Llama-3.3-70B NLAs
4. **Our side**: run v0.1.dd step_250 AV + AR v0.1 paraphrase-invariance pair locally on the matched activation — running against our own Gemma-4-E2B model
5. **LLM judge**: scored each (source, anthropic_explanation, our_explanation) on validity + specificity (1-5) + preferred. Two-judge cross-validation: Claude judge (49/50 anthropic preferred) and Gemini judge with explicit size-gap calibration in the prompt (48/49 valid preferred anthropic, 1 tie) both return the same verdict.

Full harness in source repo: `experiments/v8_nla_local/results/content_aware_eval/{neuronpedia_comparison.py, judge_neuronpedia_comparison.py, judge_neuronpedia_comparison_gemini.py}`.

## Important caveat — what this comparison is and is not measuring

The "Anthropic preferred N/N" framing is honest about what we measured but compounds two effects that the comparison cannot disentangle:

1. **Cross-NLA capability gap.** Anthropic's NLAs are full-FT + GRPO on bf16 27B/70B parameters with extensive training compute. Ours is LoRA r=80 + NF4 4-bit + 50-300 step SFT on 2B parameters at 4 GB VRAM. This is the gap we set out to characterize.

2. **Cross-model activation gap.** Anthropic's NLAs read 27B-Gemma-3 L41 / 70B-Llama-3.3 L53 activations on a given source text. Ours reads 2B-Gemma-4-E2B L23 activations on the same source text. These are different objects — different model families, different layers, different dimensionalities, and (per superposition theory) different per-neuron polysemanticity. The two activations may encode genuinely different concepts even when derived from identical source text.

When a 27B L41 activation resolves to "Hillary Clinton primary momentum" and our 2B L23 activation only resolves to broader "political-news content," some of that is **our NLA being worse at content reading** (the gap we wanted to measure) and some is **the underlying 2B L23 activation actually encoding less per-instance specificity** (an intrinsic model property our NLA can't compensate for). With current data we cannot disentangle the two.

The clean test that would disentangle these factors: train an NLA on Gemma-3-27B L41 using OUR exact recipe (LoRA r=80, NF4 4-bit, 50-step SFT, same labeled corpus extracted at L41). If L2 cross-row argmax + Δmse lift substantially on 27B-at-our-recipe, polysemanticity-at-2B-scale is the dominant factor and there's an intrinsic ceiling our NLA approaches. If they don't lift, training-stack constraints (LoRA + NF4 + step count) are the bottleneck and model size is incidental. Until that experiment runs, the honest characterization is that both factors compound and we don't know the relative weight. ~30-50 A100-hr on cloud GPU; flagged for next-grant work.

**Implication for the v0.0.1 + v0.1 release.** The v0.0.1 + v0.1 pair is **the only NLA that reads Gemma-4-E2B L23 activations.** There is no published reference NLA for this model+layer combination — Anthropic's deployed NLAs are on different models entirely. For someone interested in Gemma-4-E2B specifically, our own internal L1a/L1b/L2 metrics (the cross-row identity test, the per-claim Δmse probe, the held-out factual/gibber discrimination) are the right calibration tools — not the cross-NLA comparison against different-model NLAs. The cross-NLA comparison is informative for "what content-fidelity looks like at full-FT + GRPO + larger-model scale" but it isn't a like-for-like benchmark for our model+layer.

Full discussion in source repo: `FINDINGS.md` §F72 Addendum 11 (polysemanticity-at-scale hypothesis + the clean disentangling experiment).

## Addendum 2026-05-29 — Phase 4 GRPO at 4 GB also produces L2 = chance

Between 2026-05-25 and 2026-05-29 we ran the previously-skipped Phase 4 GRPO post-SFT step of the Anthropic NLA recipe end-to-end on the same 4 GB GTX 1650 Ti Max-Q, across **120 rollouts** under **five reward formulations** (`mse`, `contrastive_mean`, `contrastive_max`) × **three entropy regimes** (`β=0`, `0.3`, `1.0`, `0.1`) × **MSE vs contrastive AR-keepup loss**.  Five intermediate L2 readouts (at rollouts 40, 60, 80, 100, 120) all show **L2 cross-row argmax accuracy = 0.100 (chance)** on the n=10 seed=0 eval; mean margin oscillates between −0.130 and −0.151 in a noise band around −0.14 — the same range as the pre-GRPO v0.1 pair.

A qualitative inspection of the AV outputs at each checkpoint surfaces a *second* failure mode beyond L2 invariance: the rollouts at high entropy bonus (r60: entropy ≈ 1.7; r80–r120: entropy ≈ 4.0) produce **degenerate AV outputs** — random Unicode tokens at r60, whitespace at r80, mode-collapsed "evasion evasion evasion …" at r100/r120.  Only the r40 checkpoint (after MSE-reward GRPO with no entropy bonus) preserves the coherent multi-paragraph NLA-style output of the SFT v0.1 baseline, and its margin (−0.149) is statistically indistinguishable from the SFT baseline.

This adds a new empirical fact to the cross-NLA characterization above: combining (a) the 8-attempt SFT-only ceiling documented in `ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md` and (b) this 5-readout GRPO ceiling, **the cumulative 14-attempt picture spans the full Anthropic NLA recipe at 4 GB and produces L2 = chance under every tested loss + entropy + training-paradigm configuration**.  No checkpoint from the GRPO loop is shipped because none of them clear the bar of the existing v0.0.1 + v0.1 SFT release.

The proposed clean disentangling experiment (cross-model + recipe-controlled training on Gemma-3-27B L41) remains the load-bearing missing data point.  This GRPO result strengthens the case that the disentangling experiment is the right next test — the 4 GB ceiling is robust to both training paradigms, so the open question is whether the bottleneck is the **base model scale** (2B vs 27B/70B) or the **at-4 GB hardware constraint** (LoRA + NF4 + small contrast pool).

Full discussion + per-checkpoint margin/output/entropy/reward tables in source repo: `experiments/v8_nla_local/autoresearch/notes/GRPO_CEILING_FINDING_2026-05-29.md`.

## Addendum 2026-05-30 — Methodology alignment with the open-source kitft reference

A pre-submission audit was performed on 2026-05-30 to verify that the v0.0.1 + v0.1 release attempts every component of the open-source Anthropic NLA recipe (`kitft/natural_language_autoencoders`), to the extent possible on 4 GB consumer hardware. The full audit (with file-path evidence for every component) lives in the source research repo: `experiments/v8_nla_local/autoresearch/notes/METHODOLOGY_ALIGNMENT_AUDIT_2026-05-30.md`. The headline mapping:

| kitft component | Our status | Notes |
|---|---|---|
| Stage 0 — activation extraction | implemented | Schema match with `nla/datagen/stage0_extract.py`; L23/35 ≈ 2/3 matches kitft's middle-third heuristic |
| Stage 1 — doc-level split | implemented | 60/20/20 by hashed `doc_id`; no document-level leakage |
| Stage 2 — LLM-judge labeling | adapted | Two-pass labeler → auditor with explicit Dr. Chen / Dr. Otsuka persona prompts (kitft ships only the provider interface, not concrete prompts) |
| Stage 3 — training format + injection convention | implemented | Literal kitft AV/AR templates; `injection_scale = sqrt(d_model) = 39.25` matches kitft's "scaled to embed-token norm" practice |
| AV SFT | adapted | LoRA r=64 α=128 on NF4 base instead of full FT bf16; trigram-anchored injection hook + response-only loss preserved |
| AR SFT | adapted | First 18 of 35 layers + `Linear(1536, 1536)` head with `0.1 × torch.eye(D_MODEL)` identity initialization (matches kitft's load-bearing identity-init pattern with a small scaling factor) |
| Phase 4 GRPO | adapted, run end-to-end | REINFORCE + AR-keepup SFT + KL anchor + entropy bonus, on alternating AV/AR loads at 4 GB; 14 cumulative attempts, all L2 = chance (per the 2026-05-29 addendum above) |
| Round-trip cosine eval | implemented | 0.438 ± 0.054 (v0.0.1, n=42); 0.460 (v0.1, n=50) |
| Paraphrase-invariance auxiliary loss | extension (not in kitft) | Released as the AR v0.1 paraphrase-invariance variant |
| Steganography / claim-ablation eval | partial — per-claim Δmse probe only | kitft does not ship these either; they are paper-§ analyses Anthropic ran but did not open-source |
| Multi-GPU FSDP / Megatron / SGLang serving | out of scope | Impossible at 4 GB; explicitly skipped |

**Two evals that produce headline numbers in this release are not kitft-shipped metrics.** L1 (gibber vs real activation discrimination; v0.2 = +0.195 gap) and L2 (per-row identity via cross-row argmax; this release = chance) are diagnostics defined in the v8 research effort to characterize the AR's structural-projection failure mode and per-row content reading. They are not in the open-source kitft recipe. The "L2 at chance across 14 attempts" headline is therefore against the v8 internal bar — readers comparing this release to deployed NLAs via L2 should be aware that the L2 metric itself is a v8 construct, not a published kitft benchmark.

**Two deliberate (non-hardware-forced) divergences from kitft:** (1) GRPO reward formulation drift — kitft `configs/rl.sh` ships only `-mse_nrm`, while we tested 5 reward variants (MSE, contrastive_mean, contrastive_max) plus entropy regimes β ∈ {0, 0.1, 0.3, 1.0}, motivated by the structural-projector content-blindness pathology on v0.0.1; (2) brief AR-contrastive loss experiment at GRPO rollouts 108–120 that produced a worse reward and was reverted. Both are documented in `experiments/v8_nla_local/autoresearch/notes/GRPO_CEILING_FINDING_2026-05-29.md`.

**One enhancement flagged for future releases (not present here):** predict-the-mean baseline + FVE-vs-mean training-step logging. kitft logs `train/fve` per step; we report cosine and L2-argmax but no FVE-vs-mean baseline. Not a correctness issue for the current release, but useful instrumentation for future training runs.

**Audit verdict:** the v0.0.1 + v0.1 release is methodologically aligned with the open-source kitft recipe to the extent possible at 4 GB. Every kitft phase has been attempted in good faith; divergences are either hardware-forced, deliberate extensions, or covered by v8-original instrumentation. The release framing in this document, the v0.1.x trajectory README retraction, and the GRPO ceiling finding remain the load-bearing honesty surface; this audit confirms the substrate they sit on.

## Addendum 2026-05-31 — Cross-family L2 test on Gemma-3-1B: ceiling is NOT family-robust

The 2026-05-29 GRPO addendum (above) framed the open question as "whether the bottleneck is the **base model scale** (2B vs 27B/70B) or the **at-4 GB hardware constraint** (LoRA + NF4 + small contrast pool)." We have now run a within-Gemma-3-family scale-controlled test at 4 GB that partially answers this.

**Method.** Trained a fresh NLA pair on **Gemma-3-1B at L17** (proportional layer depth 0.654, matching Anthropic's deployed Gemma-3-27B at L41/62 = 0.661 and our Gemma-4-E2B at L23/35 = 0.657). Identical recipe to the v0.1 Gemma-4-E2B release: LoRA r=64 α=128 on NF4 base, AdamW-8bit lr=1e-4, grad-accum=16, 512 max length, 50 SFT steps each for AV and AR, 0.1 × torch.eye identity-init AR linear head. Hardware: same 4 GB GTX 1650 Ti Max-Q.

The training data is a cross-family JOIN: Gemma-3-1B L17 activations were re-extracted by running forward passes through Gemma-3-1B on the same 800 FineWeb-Edu source-text positions used for v0.2 Gemma-4-E2B's stage0, then joined with v0.2's existing persona+audit labels by (doc_id, n_raw_tokens). Same labels, same source text, fresh activations. The labels are model-agnostic — they describe source-text content at each position, not a property of the captured activation — so reusing them across model families is a controlled comparison.

Note that Gemma-3-4B at the same recipe was tried first and pivoted: Gemma-3-4B NF4 base alone fills 3.23 GB of 4.29 GB total VRAM, leaving 0.18 GB free — insufficient for LoRA + grads + 8-bit AdamW state (~600 MB additional needed). Gemma-3-1B is the largest within-family model that fits the standard recipe at 4 GB.

**Result.** L2 cross-row argmax accuracy on n=40 (4 seeds × n=10 each) on a TRUE held-out subset of 272 source-text positions NOT in either training set:

| Metric | Released Gemma-4-E2B v0.1 (14 attempts) | Gemma-3-1B (this experiment, n=40) |
|---|---:|---:|
| Round-trip cos | 0.460 ± 0.054 | 0.9712 ± 0.006 |
| L1 noise gap | +0.195 (v0.2 noise-hinge AR) | +0.0039 |
| **L2 argmax acc** | **0.100 (chance, every attempt)** | **0.400 (4× chance, p < 10⁻⁶)** |
| L2 mean margin | −0.149 | +0.0016 |

The cross-family Gemma-3-1B pair clears L2 = 0.40 against chance = 0.10 with one-tailed Binomial p < 10⁻⁶. **The 4 GB-LoRA-NF4 L2 = chance ceiling characterized across 14 cumulative attempts on Gemma-4-E2B is NOT robust to model family.**

**Interpretation.** The most likely explanation, given the controlled recipe + similar proportional layer depth, is that Gemma-3-1B's L17 residual stream encodes per-row content in a form more amenable to the LoRA-NF4 AR's reconstruction than Gemma-4-E2B's L23 residual stream does. Consistent with the polysemanticity-at-scale hypothesis previously documented here — but now with the added wrinkle that the family-level architectural differences between Gemma-3 (text-only, vanilla decoder) and Gemma-4 (multimodal, per-layer input norm + matformer-style multi-resolution attention not present in Gemma-3) also matter, not just scale alone. Round-trip cos = 0.971 and L1 noise gap = +0.004 reflect intrinsic Gemma-3-1B L17 cosine concentration on the unit sphere (the AR is STILL structural-projector-dominant), not improved per-row content fidelity per se. But the within-distribution input-specific signal in the reconstruction is enough to discriminate row identity at 4× chance, which the Gemma-4-E2B AR never achieved.

**Confounders and limitations of the conclusion.**

1. **Cross-family training data was smaller** (360 AV rows vs released v0.1's 4734; 168 AR rows vs released v0.1's ~1700). However, the v0.1.0 11K-row corpus on Gemma-4-E2B also hit L2 = chance, so corpus scaling is unlikely to be the load-bearing factor.
2. **Cross-family training was 50 steps**; released v0.1 trained 50–300 steps. The Gemma-4-E2B ceiling held at every step count from 50 to 300, so longer training is unlikely to be the load-bearing factor.
3. **The labels were written for Gemma-4-E2B's activations at L23**, then reused for Gemma-3-1B at L17. A fresh persona+audit relabeling on Gemma-3-1B activations would isolate this confound.
4. **Initial eval at seed=0 on the full 800-row replay pool was contaminated** by 4-of-10 overlap with the AV training set (L2 = 4/10 = 0.400, identical to the training-overlap count, consistent with memorization). All reported numbers above use the 272-row true-held-out subtraction; the multi-seed verdict is robust.

**Implications for the released v0.0.1 + v0.1 Gemma-4-E2B pair.**

- The "L2 = chance across 14 attempts" finding documented for Gemma-4-E2B specifically remains correct and is not invalidated by this addendum. The release ceiling characterization stands for Gemma-4-E2B at this hardware scale.
- The "4 GB-LoRA-NF4 ceiling is robust to model family" framing previously hedged in the 2026-05-29 addendum is now empirically rejected. Model family does matter.
- The recommended next-step experiment from the 2026-05-29 addendum — cross-model recipe-controlled training on Gemma-3-27B — remains the gold-standard disentangling experiment and is now better-motivated: Gemma-3-1B at 4 GB clears L2; Gemma-3-27B with our recipe (on adequate hardware) would test whether the lift extrapolates linearly with within-family scale, isolating "family vs scale vs hardware" decisively.

Full audit + per-seed eval JSON + training logs + checkpoints in source research repo: `experiments/v8_nla_local/autoresearch/notes/CROSS_FAMILY_GEMMA3_1B_2026-05-30.md`.

## Acknowledgments

Thanks to [Neuronpedia](https://www.neuronpedia.org/) for the public NLA API that made this calibration possible. Thanks to Anthropic / Kit Fraser-Taliente et al. for the open-source [NLA methodology and reference checkpoints](https://transformer-circuits.pub/2026/nla/).
