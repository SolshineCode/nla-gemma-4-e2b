# Accuracy Collapse Limitations Root Causes Hypothesis

**Document type:** post-mortem / root-cause analysis
**Subject:** v0.0.1 Gemma-4-E2B NLA pair AV-side template collapse
**Author:** Caleb DeLeeuw (SolshineCode)
**Date:** 2026-05-12
**Status:** Hypothesis, partially verified

## What this document is

The v0.0.1 NLA pair achieves round-trip cosine similarity of 0.438 ± 0.054 on n=42 held-out activations, with 100% of evaluated rows clearing the 0.30 noise floor. Those numbers are real. But qualitative inspection of the per-row eval outputs revealed a problem the cos metric does not surface: the AV (Actor) half of the pair produces a small, templated set of explanations rather than per-row faithful descriptions of the activations it receives.

This document records what we found, the proposed root causes in rough order of contribution, and the hypothesis about which intervention fixes which cause. It is preserved as a research note for future versions of this work and for any third party reproducing the methodology.

## The empirical finding

On the n=42 evaluated rows from `results/round_trip_v0_n50.json`:

| Granularity | Unique patterns observed |
|---|---|
| Full explanation strings | **20 / 42** (52% exact-duplicate rate) |
| First 300 characters | 20 |
| First 200 characters | 17 |
| First 160 characters | 15 |
| First 120 characters | 9 |
| First 80 characters | **4** |
| First 40 characters | **1** (every output begins `[Immediate semantic expectations: The text discusses`) |
| Generated length | 298-299 chars on every single row; max_new_tokens hit on every generation |

The four opening stems at the 80-character granularity collapse all 42 evaluations into four genre buckets:

- `The text discusses a legal case` (34 / 42 = 81%)
- `The text discusses a protest`
- `The text discusses a new feature`
- `The text discusses a new policy`

Spot check: `doc_00000001` is verifiably **Hillary Clinton at a campaign rally** in The Washington Post text, not a legal case. The AV labels this row with the "legal case" stem and the matched AR round-trips it back to roughly the correct activation region anyway, producing cos = 0.370 to 0.532 across the three positions sampled from that doc. The cos number is real. The explanation is not.

## Root causes, in rough order of contribution

### 1. Severely under-trained AV

The v0.0.1 AV was trained for **55 SFT steps** on **2,548 (text, activation) rows** at effective batch size 16. That is roughly 880 batches across 2,548 rows, well under one effective epoch of meaningful gradient signal per row.

Anthropic's NLA paper does not publish exact corpus and step counts for their 7B variants, but their reference setup uses orders of magnitude more training (typical SFT runs for explanation models target 1,000-10,000 steps on corpora of 10K-100K rows). With v0.0.1's training budget, the AV converges to a small set of high-probability output trajectories that minimize cross-entropy on the training distribution and stops exploring.

**This is the predicted small-model failure mode under aggressive descope.** It is the load-bearing cause in our hypothesis.

### 2. Training labels themselves have low diversity

v0.0.1 was labeled by gpt-4o-mini via a single prompt template (the verbatim Fraser-Taliente et al. INSTRUCTION). gpt-4o-mini exhibits a strong tendency to produce templated outputs across similar inputs, particularly when given a strict output-format schema like the NLA `[Immediate semantic: ...] [Narrative momentum: ...] [Final token: ...]` structure. The resulting 2,548-row SFT corpus thus has limited diversity in label style: the AV inherits whatever style space the labeler occupied.

The v0.1.x interim AV trained on **Gemini CLI + persona+audit pipeline** labels (Dr Marisol Chen labeler + Dr Riley Otsuka auditor) shows **55 unique opening patterns across 97 evaluated rows** at equivalent round-trip cos (0.441 vs 0.438). This is the strongest empirical signal we have for which lever matters: label diversity, not SFT step count alone. v0.1.x interim used only 200 SFT steps and still produced an order of magnitude more per-row variation than v0.0.1's 55 SFT steps did.

### 3. SFT cross-entropy loss has no diversity penalty

The training objective is standard autoregressive cross-entropy on the labeled explanations. Tokens that match the label are rewarded; tokens that don't are penalized. If the labels themselves are templated (cause #2), low loss is achieved by outputting templates. There is nothing in the SFT signal that pushes the AV to generalize beyond the modes present in the training labels or to maintain per-row diversity at inference time.

Anthropic's full methodology includes a GRPO RL Phase 4 (their v0.2.0-equivalent) that adds a reward signal beyond cross-entropy. v0.0.1 has not implemented Phase 4. Whether RL would on its own fix template collapse without first scaling the SFT corpus is an open question and depends on the design of the reward function. The Anthropic paper's RL phase rewards round-trip faithfulness, which by our finding does NOT directly penalize template collapse (causes 1-3 produce template-collapsed outputs that still achieve respectable round-trip cos).

### 4. Round-trip cosine similarity does not reveal the failure mode

The published round-trip cos = 0.438 ± 0.054 is a joint AV+AR metric. The matched AR has learned during its own SFT to map each of the 4 opening genre stems back to a broad region of activation space. This means that even when the AV produces the wrong template for a given activation (legal-case for Hillary Clinton), the AR can still reconstruct a vector that lies in the same general region as the original.

This is structural, not a bug in our eval: round-trip cos measures **closure of the pair as a system**, not **per-row AV faithfulness**. We caught the diversity problem only on qualitative inspection of the per-row `explanation` field in the result JSON. Future eval pipelines should include an explicit per-row diversity metric (proposed: unique-templates-per-100-rows at the 80-char, 200-char, and full-string granularities) alongside cos so the dissociation is visible up front.

### 5. Held-out OpenWebText eval set is partially homogeneous

The v0.0.1 eval set was drawn from the same OpenWebText pool as the training corpus, doc-keyed split. OpenWebText contains a heavy news-and-commentary slant, and many activations within it likely cluster in regions that loosely correspond to legal/political/policy/product genres. The AV's 4-template output is therefore not random nonsense; it picks one of four genuinely-present semantic regions and slots the activation into it.

This is the weakest of the five causes. The "legal case" template returning for a Hillary Clinton rally activation is proof that the bucketing is wrong even within OpenWebText, but the eval set's homogeneity makes the failure mode less visible than it would be on a diverse held-out set (e.g., a sample including Wikipedia, arXiv, code, dialogue). v0.1.x evaluations should run on a deliberately heterogeneous held-out set so any future template collapse is more obvious.

## Hypothesis: which interventions fix which causes

| Intervention | Causes addressed | Empirical evidence |
|---|---|---|
| Scale SFT to 3,000-5,000 steps on full diversified corpus | 1 (primary), 5 (secondary) | None yet at v0.1.0 scale; v0.1.x interim at 200 steps shows partial improvement |
| Diversify the labeler (Gemini + persona+audit, multi-labeler corpus) | 2 (primary) | **Strong: v0.1.x interim shows 55 patterns vs v0.0.1's 4 at same cos** |
| Add per-row template-diversity metric to eval | 4 | Not yet implemented |
| Replace OpenWebText-only eval with heterogeneous held-out set | 5 | Pending |
| GRPO RL Phase 4 with diversity-aware reward | 3 | Not implemented; design open |
| Train on 7B or larger base model via cloud GPU | 1 (capacity ceiling check) | Pending Mistral-7B / OLMo-7B second-model release |

The hypothesis is that causes 1 and 2 together account for the bulk of the failure, and that v0.1.0 (full diversified corpus + 3,000-5,000 SFT steps) will materially improve per-row diversity while keeping cos in the 0.4+ range. If v0.1.0 lands and the diversity gain is small, cause 3 becomes load-bearing and Phase 4 RL becomes necessary. If v0.1.0 lands with strong diversity but cos drops below 0.40, the dissociation seen at v0.1.x interim was a fluke and we need to revisit the eval methodology.

## What this artifact still is, despite the limitation

v0.0.1 is not a usable per-row interpretability tool. It is a methodology infrastructure release with under-trained baseline adapters. The pipeline itself, the descope choices, the multi-labeler infrastructure, the persona+audit labeling pipeline, the honest-accuracy training-trend convention, the published HF datasets, and the reproducibility chain are all independently valuable as research outputs. They were not invalidated by the template-collapse finding. The trained adapters specifically are the part that's under-resourced and exhibits the expected small-model failure mode.

This document exists to make that distinction unambiguous in the record.

## Cross-references

- Per-row eval JSON: `experiments/v8_nla_local/results/round_trip_v0_n50.json` (full 42-row content with per-row cos and explanation)
- v0.1.x interim per-row eval JSON: `experiments/v8_nla_local/results/round_trip_v0_1_0_interim_n100.json` (97 rows, 55 unique patterns)
- Model card: `experiments/v8_nla_local/release/v0_0_1/README_HF.md` (Limitations section now references this document)
- Public release model card: `nla-gemma-4-e2b/MODEL_CARD_AV.md`
- Honest-accuracy convention: `CLAUDE.md` Research Interpretation Guardrails section

## Next steps that would materially change this assessment

1. Train matched v0.0.x AR on the Claude Haiku persona+audit labels (corpus already exists, 696 rows). Run round-trip eval on the matched Option B pair. If diversity improves at v0.0.x corpus scale just from changing the labeler, that strengthens cause #2 as the primary lever.
2. Complete v0.1.x ar_sft labeling (watchdog in progress, ~16% complete) and train the full v0.1.0 NLA pair. Run round-trip eval with the new per-row diversity metric. This is the load-bearing experiment.
3. Implement the per-row diversity metric in `eval_round_trip.py` and back-fit it to all existing eval JSONs to surface the dissociation historically.
4. Begin the second-model NLA (Mistral-7B or OLMo-7B on rented A100) to test whether scale + diverse labels + matched pair produces both cos > 0.5 AND per-row diversity > 50 patterns per 100 rows at the 7B base model size.
