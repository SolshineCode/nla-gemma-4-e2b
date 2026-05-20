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

## Acknowledgments

Thanks to [Neuronpedia](https://www.neuronpedia.org/) for the public NLA API that made this calibration possible. Thanks to Anthropic / Kit Fraser-Taliente et al. for the open-source [NLA methodology and reference checkpoints](https://transformer-circuits.pub/2026/nla/).
