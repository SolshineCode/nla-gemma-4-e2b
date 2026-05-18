# Accuracy Collapse Limitations Root Causes Hypothesis

**Document type:** post-mortem / root-cause analysis
**Subject:** v0.0.1 Gemma-4-E2B NLA pair AV-side template collapse
**Author:** Caleb DeLeeuw (SolshineCode)
**Date:** 2026-05-12
**Status:** Hypothesis, partially verified

> ## ⚠ CORRECTED 2026-05-16 — significant retraction below this document's "H24 bf16+inj=80000" addendum (if present)
>
> Subsequent analysis identified that the `injection_scale` parameter used in v0.1.x trial runs after 2026-05-15 was set 510–2038× larger than the Gemma-4-E2B token-embedding norm (measured 39.25). The injected activation vector was out-of-distribution to the transformer, the AV learned to ignore the injection slot, and template collapse appeared from training step_10 regardless of any other lever.
>
> The phenomenon described in *this* document (v0.0.1's AV-side template collapse) is real and unchanged — v0.0.1 was trained at the in-distribution default `injection_scale = sqrt(d_model) = 39.2`, coincidentally matching the embed norm. The retraction applies to subsequent reasoning that "fixing this collapse requires hardware beyond 4 GB" — the v0.1.bb experiment at corrected `injection_scale=39` (2026-05-16) showed that the injection_scale bug caused a specific failure mode (empty-output collapse, v0.1.aa) but NOT the broader content-blindness phenomenon. The broader phenomenon is more likely a step-count + effective-batch-size ceiling than a hardware-impossibility.
>
> Full retraction with audit table and predictions/refutations: `FINDINGS.md §F72` in the source research repo (private; DM for access).
> Process retrospective on how the AI-introduced bug propagated for 8 days: `notes/AI_RESEARCHER_LESSON_2026-05-16_injection_scale_hallucination.md` in the source research repo.

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

---

# Investigation Report: 6-Hypothesis Team Findings (2026-05-12)

## Method

After authoring the five-cause hypothesis above, a team of six subagent / direct investigations ran in parallel and sequence against the v0.0.1 AV+AR pair. Three CPU-only data audits, three GPU experiments. Each tested a distinct hypothesis. Results consolidated and synthesized below.

| Agent | Hypothesis | Verdict | Mechanism |
|---|---|---|---|
| H3 | gpt-4o-mini labels are themselves templated | **REFUTED** | 100% unique label strings, 957 unique "discusses X" fillers, mean label length 542 chars |
| H4 | Activations cluster naturally into 4 groups | **REFUTED** | Mean pairwise cos 0.24, effective rank ~17, silhouette < 0.12 at all k. Activations are diffuse; templates do not align with structure |
| H6 (open) | Surface new root causes | **Surfaced H9, H10, H11, H12** + storage-truncation bug | See below |
| H1 | Greedy decoding masks AV diversity | **PARTIALLY SUPPORTED but more diagnostic finding** | Sampling produces 100% unique strings vs greedy's 5/8, but **all sampled outputs are topically wrong** |
| H11 | max_new_tokens=120 truncates generation | **REFUTED** | Greedy naturally terminates at 520-605 chars regardless of cap (cap=120 and cap=250 produce identical-length outputs) |
| H2 | AV ignores injected vector | **PARTIALLY SUPPORTED** | ZERO vector → 100% empty output (10/10). REAL vs RAND produce same-magnitude diversity (6 vs 7 unique). Direction matters, but mapping is essentially arbitrary |
| H5/H8 | AR has positional shortcuts; cos doesn't measure faithfulness | **STRONGLY CONFIRMED** | AR produces cos 0.405 from EMPTY input. AV explanation contributes mean 0.024 cos over baseline. The published cos 0.438 is ~95% structural AR projection, ~5% explanation-dependent |

## Storage truncation bug (the meta-finding)

`eval_round_trip.py:195` stored `"explanation": explanation[:300]` in the published v0.0.1 eval JSON. Every analysis based on that JSON read 300-char prefixes, not full outputs. The published "every output is exactly 298-299 chars" reading was an artifact of this storage cap, not a model behavior. The model actually emits 520-605 chars per row under greedy decoding. The "4 opening template stems / 81% legal case" finding was based on inspecting the prefix of truncated strings. **The truncation bug had been propagated into the public artifacts and shaped the original 5-cause hypothesis incorrectly.** Fix landed in this investigation pass.

## The H5 result is the load-bearing finding

The remaining 5 root-cause hypotheses (H1-H12) were chasing a phenomenon — template collapse — that turns out to be secondary. **The primary failure mode is structural in the AR**, not in the AV's generation diversity. Specifically:

For 10 rows from rl.parquet, we measured AR-reconstruction cos under four explanation conditions:

| Condition | Mean cos | Median | Std | All above 0.30 floor |
|---|---|---|---|---|
| **AV_OUT** (the actual AV-generated explanation) | **0.4292** | 0.4063 | 0.0501 | 10/10 |
| GIBBER (random tokens / nonsense words) | 0.4135 | 0.4004 | 0.0432 | 10/10 |
| RANDOM (unrelated Wikipedia sentences) | 0.4045 | 0.3880 | 0.0432 | 10/10 |
| EMPTY (literally empty string) | 0.4051 | 0.3929 | 0.0424 | 10/10 |

Mean within-row range across 4 conditions: 0.030. AV_OUT - EMPTY mean delta: +0.024 ± 0.021. AV_OUT - GIBBER mean delta: +0.016. **The "explanation" contributes about 5% relative to the noise-floor headroom.** The remaining 95% of cos value is determined by the AR's structural projection from any text input toward "somewhere in OpenWebText L23 activation space."

This means:
- The published headline cos 0.438 is principally a measurement of AR's content-independent projection, not a measurement of AV explanation faithfulness.
- The 0.30 noise-floor framing was meaningless: feeding the AR nothing at all produces cos that clears the floor 10/10 times.
- Any reproduction with arbitrary text would have shown roughly the same headline number.

## The H2 result clarifies what the AV does and doesn't do

| Injection | Empty rate | Unique full strings | Notes |
|---|---|---|---|
| REAL (the actual L23 activation) | 2/10 | 6/10 | Templates: 8 legal-case-stem + 1 new-feature-stem |
| ZERO (zero vector) | **10/10** | 1 (all empty) | Deterministic empty output |
| RAND (gaussian unit vector × sqrt(d_model)) | 1/10 | 7/10 | More diverse than REAL |
| REAL repeated 3x on row 0 | 3/3 empty | 1 | Greedy is deterministic |

Random gaussian vectors at the right magnitude produce **more** template variation than the actual learned activations. The AV is sensitive to injection direction (ZERO vs anything else affects whether output appears, and real vs random can pick different templates), but the mapping from direction to template is essentially arbitrary — random noise activates the same template space as real activations.

Combined with the topic-mismatch observation (Hillary Clinton activations → "legal case" template under greedy or "future of work / merger / police investigation" under sampling): **the AV has learned to be sensitive to injection but has not learned to map activation content to faithful explanations**. Activation injection picks which arbitrary template gets emitted; the activation's actual semantics are not preserved.

## The H1 result clarifies what sampling does

10 rows × 4 generation configs (greedy/sampling × max_new_tokens 120/250):

| Config | n_nonempty | Unique full strings | Unique first-80 | Unique first-200 | Mean length |
|---|---|---|---|---|---|
| A (greedy, max_new=120) | 8/10 | 5/8 | 2/8 | 5/8 | 445 chars |
| B (greedy, max_new=250) | 8/10 | 5/8 | 2/8 | 5/8 | 449 chars |
| C (sampling T=0.7 top_p=0.9, max_new=120) | 9/10 | 9/9 | 9/9 | 9/9 | 522 chars |
| D (sampling, max_new=250) | 9/10 | 9/9 | 9/9 | 9/9 | 544 chars |

Sampling delivers 100% lexically-unique outputs vs greedy's 5/8. But qualitative inspection of the sampled outputs reveals **all are topically wrong**: same Hillary Clinton activation produces "future of work" (row 0), "new policy / 100" (row 1), "successful merger" (row 2), "police investigation" (row 3). The diversity is real at the surface form, but the outputs are not source-aligned.

H11 (max_new_tokens truncation) is refuted: B's outputs are nearly identical to A's (length difference 4 chars at cap difference of 130 tokens), meaning greedy is naturally terminating well before either cap. The 298-299 chars in the published eval was 100% from the storage truncation bug.

## Synthesis: what is actually going on in v0.0.1

Putting H1 / H2 / H3 / H4 / H5 / H6 together:

1. **The AV has learned the NLA template form** (`[Immediate semantic: ...] [Narrative momentum: ...] [Final feature: ...]`) at 86%+ prefix-stem reliability across training. This is the easy lesson and it was successfully learned.

2. **The AV has not learned to condition the body content on the injected activation.** It treats the activation injection as a soft-selector between a small set of unconditional-prior continuations (greedy → 4 templates), or as a noise source (sampling → diverse but topically unrelated content).

3. **The AR has not learned to use the explanation text.** It learned to project from "any English text under the `Summary of <text>...</text> <summary>` template" toward a constant region of L23 activation space. This region is moderately correlated with actual L23 activations from OpenWebText (mean cos ~0.40) because both the projection and the held-out activations sit in the same broad subspace of the residual stream.

4. **The "round-trip" eval measures item (3), not items (1) and (2).** Cos 0.438 is structural projection, not faithfulness. EMPTY string in, cos 0.405 out.

5. **Both halves are under-trained at this scale.** Effective training exposure for the AV was ~12% (220 rows seen once at grad-accum-adjusted budget). The AR's linear head was initialized to 0.1× identity and only fine-tuned for similar effective steps. Neither half has had enough gradient signal to learn the load-bearing mapping (vector→language for the AV, language→vector for the AR).

The 5 original hypotheses are re-ranked by this investigation:

| Original cause | New rank | Notes |
|---|---|---|
| 1. Under-trained AV | Still primary | Confirmed; ~12% data exposure |
| 2. Low-diversity labels | Demoted to secondary | Labels themselves are diverse (H3); template-prefix is the inherited part |
| 3. No SFT diversity penalty | Demoted to secondary | Cross-entropy on diverse targets should have worked; under-training is what kept the model on the modal prior |
| 4. Cos doesn't reveal failure mode | **Promoted to PRIMARY** | H5 shows cos is dominated by structural AR projection, not faithfulness |
| 5. OpenWebText eval homogeneous | Refuted | Activations are diffuse (H4); eval homogeneity isn't the driver |

**New primary causes:**
- **AR is content-blind** (H5/H8 confirmed): the AR has learned a near-constant projection that the explanation text barely modulates.
- **AV does not condition on activation content** (H2 partial + H1 qualitative): injection direction picks the template, not the content.
- **Both halves are severely under-trained** (H1/H12 confirmed): ~12% effective training exposure makes the above two outcomes the predicted small-model failure mode.

## Implications for the public release

The v0.0.1 release on HuggingFace + the bundled `nla-gemma-4-e2b` repo describes the cos 0.438 number with extensive honesty framing about per-row template collapse. The H5 finding tightens that further: cos 0.438 does not measure explanation faithfulness in any meaningful sense — feeding the AR random Wikipedia sentences or nothing at all produces cos in the same range. The artifact's honest framing should now say:

> v0.0.1 is a methodology-pipeline release. The trained AV and AR were under-resourced relative to the methodology's requirements and do not produce faithful per-row activation explanations. The round-trip cosine similarity of 0.438 measured on this release is principally a structural artifact of the AR's content-independent projection (mean cos 0.405 from empty-string input on the same eval set) rather than a measurement of explanation faithfulness.

Public artifacts are being updated in this pass to reflect this.

## What we still don't know

- **Does scaled training fix this?** The under-training hypothesis predicts yes, but the H5 finding suggests both halves may need re-architecting around a stronger language→activation training signal, not just more steps. Anthropic's RL Phase 4 is presumably the intended mechanism but is not implemented here.
- **What is the H5 baseline at 7B scale?** If we could replicate H5 against Anthropic's published Qwen-7B NLA, we'd know whether content-blind projection is a small-model phenomenon or a generic property of the round-trip-cos metric at all scales.
- **Does a stronger AR alone help?** Training a better AR (e.g., on Haiku persona+audit ar_sft labels — corpus exists at 696 rows) is a feasible local experiment to isolate the contribution of the AR side. Approximately 3-6 GPU-hours on the 4 GB card.

## Eval methodology recommendations for v0.1.0+

1. **Always also report empty-explanation cos as a baseline.** If `cos(AV(v), AR(AV(v))) - cos(AV(v), AR(""))` is near zero, the round-trip cos is not measuring faithfulness regardless of its absolute value.
2. **Report `unique-templates-per-100-rows` at multiple granularities** (40, 80, 120, 200, full).
3. **Remove the `[:300]` storage truncation in the result JSON.** Save full per-row explanations.
4. **Include topic-alignment sanity check.** A simple LLM-judge pass over per-row (source text, AV explanation) pairs gives a faithfulness signal independent of round-trip cos.
5. **Add the H2 injection-fidelity test as a recurring diagnostic.** A future AV should produce empty under ZERO injection (as v0.0.1 does, which is the correct behavior) but should produce *content-distinguishable* outputs under REAL vs RAND vectors.

## Files and reproducibility

All raw data, scripts, and per-row outputs are committed in `experiments/v8_nla_local/results/`:
- `round_trip_v0_n50.json` (original 42-row eval; explanations truncated at 300 due to legacy bug, fixed)
- `h1_h11_results.json` (10 rows × 4 decoding configs; full untruncated explanations)
- `h2_injection_fidelity_results.json` (10 rows × 3 injection conditions + 3 repeats)
- `h5_ar_gibberish_results.json` (10 rows × 4 input-text conditions)
- `h13_new_ar_results.json` (10 rows × 4 conditions, against Haiku-relabeled retrained AR)
- `h14_training_scale_trend_results.json` (5 AR checkpoints × 10 rows × 4 conditions, training-scale trend)
- `h3_analysis.py`, `h1_h11_eval.py`, `h2_injection_fidelity.py`, `h5_ar_gibberish.py`, `h13_new_ar_eval.py`, `h14_training_scale_trend.py` for reproduction

---

# Addendum 2 (2026-05-13): H13 and H14 ablations — Both refuted

After the original 5-experiment investigation, two further GPU ablations on the local 4 GB GTX 1650 Ti Max-Q tested whether either label quality or step count, within the v0.0.x compute regime, could improve content-sensitivity. Both came back as clean negative results.

## H13: Better AR labels alone, same step budget

Trained a new AR ("Option B AR") on the Claude Haiku persona+audit ar_sft corpus (696 rows) for 60 effective SFT steps (vs v0.0.1's gpt-4o-mini-labeled AR at similar step budget). Re-ran the H5 ablation against the new AR.

**Result:** AV_OUT − EMPTY mean delta moved from baseline +0.0242 to **+0.0174** — slight regression on the mean. Std delta more than doubled (0.022 → 0.049). Two rows (1 and 8) showed strong content-sensitive signals (+0.096 / +0.098, 4-15× larger than baseline), but three rows (6, 7, 9) showed WORSE content-sensitivity; row 6 went actively negative (−0.063, explanation hurt reconstruction). Better labels at v0.0.x scale created more *variability* in content-sensitivity, but did not move the mean.

## H14: More training steps, same labels

Re-ran the H5 ablation against 5 saved AR checkpoints from H13 corresponding to 20, 30, 40, 50, and 60 effective training steps.

| Effective steps | AV_OUT mean cos | EMPTY mean cos | Mean delta |
|---|---|---|---|
| 20 | 0.4226 | 0.4048 | +0.0179 |
| 30 | 0.4508 | 0.4329 | +0.0179 |
| 40 | 0.4551 | 0.4343 | +0.0207 |
| 50 | 0.4533 | 0.4350 | +0.0183 |
| 60 | 0.4409 | 0.4235 | +0.0174 |

**Mean delta is dead flat across the range** (0.0174-0.0207, std across checkpoints ~0.0012). Training more steps does not help. Std delta grows monotonically (0.014 → 0.046) — the AR becomes more variable, not more reliable. Absolute AV_OUT cos peaks at step_000030 (0.455) then plateaus / declines, suggesting mild overfitting on the small 696-row corpus past step 30.

## Combined finding (H13 + H14)

Neither label quality nor step count, alone or in combination within the v0.0.x compute regime, moves the explanation-conditional cos delta. The +0.018-0.024 range is a **regime-bounded ceiling** at the v0.0.x descope (4 GB GPU + ~700-2000-row corpus + ≤60 SFT steps).

This collapses the original 5 root causes into one load-bearing constraint: **insufficient training compute at this descope**. The five original causes (under-trained AV/AR, label diversity, no SFT diversity penalty, cos doesn't measure faithfulness, OpenWebText eval homogeneity) are reorganized:

| Original cause | Post-H13/H14 status |
|---|---|
| Under-trained AV/AR at v0.0.x scale | **PRIMARY** (confirmed) |
| Cos doesn't measure faithfulness (structural projection dominates) | **CO-PRIMARY** (H5; reinforced by H14 — structural cos rises with training while content-conditional component is flat) |
| Label diversity alone | **REFUTED** (H13) |
| Step count alone | **REFUTED** (H14) |
| No SFT diversity penalty | Secondary (untested directly) |
| OpenWebText eval homogeneity | Refuted (H4) |

## What still might break the ceiling

- **Substantially larger corpus** (v0.1.x is 4,734 rows, 6.8× larger; untested at v0.0.x step scale)
- **Substantially more SFT steps** (cloud GPU could support 3,000-5,000 steps, 50× current budget; untested)
- **Larger base model** (Mistral-7B / OLMo-7B at the methodology's intended scale)
- **Different objective** (Phase 4 RL with faithfulness-aware reward)

H13 and H14 together establish that these are necessary, not just nice-to-have.

---

## Addendum 2026-05-14: H15 + diagnostic evals + comparison to upstream Anthropic NLA

This addendum is **append-only** per the project's results-discipline directive. Earlier H1-H14 analysis is preserved verbatim above; the new findings below extend and partially reframe it.

### Provenance

- H15 (rank-scale ablation): `experiments/v8_nla_local/results/template_collapse_investigation/h15_step_000200_results.json`. Run log: `experiments/v8_nla_local/logs/h15_step_000200_run.log`. Script: `h15_cheap_path_eval.py`. Merged in PR #106 on 2026-05-14.
- Step_200 diagnostic evals (mode collapse + content match): `experiments/v8_nla_local/results/template_collapse_investigation/step_200_av_outputs_30rows.json`, `step_200_mode_collapse.json`, `step_200_content_match_judge.json`, `step_200_diagnostic_summary.json`. Notes: `notes/STEP_200_DIAGNOSTIC_EVALS_2026-05-13.md`. Content-match judge: claude-sonnet via `claude -p` subprocess (Claude Code subscription credits). PR #107.
- r=80 LoRA capacity bump (partial trajectory, in-progress): `experiments/v8_nla_local/checkpoints/av_v0_1_x_y_r80/step_000050/`, `step_000100/`, `step_000150/`. Run log: `logs/av_v0_1_y_r80_train.log`. Branch `session/lora_rank_bump_r128`.
- Upstream Anthropic NLA injection-scale data points: pulled 2026-05-14 from HuggingFace sidecars at `kitft/nla-qwen2.5-7b-L20-av/nla_meta.yaml`, `kitft/nla-gemma3-12b-L32-av/nla_meta.yaml`, `kitft/nla-gemma3-27b-L41-av/nla_meta.yaml`. Source code reference: `kitft/natural_language_autoencoders/nla/schema.py:normalize_activation` and `nla/train_actor.py:526` (`self._nla_vectors = normalize_activation(popped, self._nla_cfg.injection_scale)`).

### H15: v0.1.x cheap-path scale-up (4,734-row corpus, 200 SFT steps, r=64)

Trained a fresh AV on the 6.8× larger Gemini persona+audit corpus for 200 steps, then ran the same H5-style 4-condition ablation (AV_OUT / RANDOM / GIBBER / EMPTY) against the same v0.0.1 AR.

- AV_OUT mean cos: 0.4258
- EMPTY mean cos: 0.4051
- **AV_OUT − EMPTY delta: +0.0207** (std 0.0202, n=10)
- vs v0.0.1 baseline delta +0.0242, v0.0.2 baseline +0.0173

**Result:** 9× more effective row exposures than v0.0.1 (3,200 vs ~220) does NOT move the content-sensitivity delta out of the v0.0.x noise band. The original "Substantially larger corpus" lever, on its own, is REFUTED at this corpus size.

### Diagnostic A — mode collapse at step_000200 (n=30 rows)

Generated 30 AV outputs from `data/stage1/rl.parquet` rows 0-29, computed unique-prefix statistics.

- Unique 30-char prefixes: 10/30
- Unique 60-char prefixes: 11/30 (strict-threshold verdict: AMBIGUOUS, between 8 and 20)
- Unique full strings: 23/30
- Mean pairwise bigram Jaccard: 0.167

Top 60-char prefixes:

| Prefix | Count |
|---|---|
| `The model tracks the "unresolved" status of the "unresolved"` | **15 / 30 (50%)** |
| `The model tracks the "no longer" temporal adverbial phrase,` | 4 / 30 (13%) |
| (empty string) | 3 / 30 (10%) |
| (eight other prefixes, each 1 / 30) | 8 / 30 |

The top 3 patterns cover 73% of rows. The strict-threshold AMBIGUOUS verdict is a technicality — in practice this is strong mode collapse, dominated by ~3 templates and a long tail of single-occurrence variations.

### Diagnostic B — content match via Claude (claude -p / Sonnet) on the same 30 rows

For each (source_text, AV_explanation) pair, asked claude-sonnet (via `claude -p` subprocess on Claude Code subscription credits, NOT API billing) to score the AV's explanation against the source text on a 1-5 rubric:
- 5: strong specific match
- 4: partial match
- 3: generic but not contradicted
- 2: weak / mostly mismatched
- 1: clear mismatch

Result: **mean score 1.23 / 5**, distribution {1: 25, 2: 4, 4: 1}. **83% of rows judged clear mismatch.** Only one row (row 18, doc_00000019, "negative polarity" template) scored a 4.

### Combined verdict from H15 + Diagnostics A & B

The v0.1.x cheap-path AV at step_000200 produces text that is **syntactically diverse enough to defeat the strict mode-collapse threshold** but **semantically disconnected from source content** in 83% of cases. The +0.0207 H15 delta is consistent with this: the AV's outputs differ enough between rows to nudge round-trip cos slightly above EMPTY, but not enough to give the AR a real content-conditional signal.

This is a stronger result than H13/H14 because it directly demonstrates the AV is **not interpreting the activation** — it's producing pre-baked templates with minor stylistic variation regardless of input.

### H17: r=80 LoRA capacity bump (in progress, partial through step_150)

Original plan was r=128 alpha=256 (3.9× more trainable params than the cheap-path's r=64 / 50M). r=128 OOM'd at optimizer-state init on the 4 GB card; r=96 also OOM'd. r=80 alpha=160 (120M trainable params, 2.4× more than r=64) is the largest LoRA rank that fits.

Loss trajectory so far (max_length=384 vs cheap-path's 512 for VRAM headroom; same corpus, same lr=1e-4, same effective batch 16):

| Step | r=64 (v0.1.x cheap-path) | r=80 (v0.1.y, this run) |
|---|---|---|
| 1 | 3.74 | 3.71 |
| 50 | 2.38 | 2.49 |
| 100 | 2.24 | 2.50 |
| 150 | 2.25 | 2.25 |

By step 150, r=80 has converged to within 0.001 of r=64 cheap-path's loss at the same step. **Mid-trajectory signal: the rank bump within the 4 GB feasible range does NOT move the loss floor.** Both runs are converging onto the same 2.20-2.30 plateau.

CAVEAT: r=80 < the original r=128 target. Cannot rule out that a substantially larger rank (3.9× or higher) would move the floor — that requires bigger hardware (4090 rental). Within the rank range we CAN test on this card, capacity is not the bottleneck.

### H18: Injection-scale mismatch with the upstream Anthropic NLA (LIKELY ROOT CAUSE) ⚠

The Anthropic upstream NLA training repo (`kitft/natural_language_autoencoders`, accompanying the Transformer Circuits 2026 post) uses a fixed `injection_scale` hyperparameter that rescales every activation vector to a target L2 norm before injecting it at the embedding layer (`nla/schema.py:normalize_activation`, `nla/train_actor.py:526`).

We use `injection_scale = sqrt(d_model) = sqrt(1536) ≈ 39.2`.

Upstream values, pulled from the released model sidecars:

| Base model | d_model | sqrt(d_model) | Upstream `injection_scale` | Ratio vs sqrt |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 3584 | 59.9 | **150.0** | 2.5× |
| **Gemma-3-12B-IT** | **3840** | **62.0** | **80,000.0** | **1290×** |
| **Gemma-3-27B-IT** | **5376** | **73.3** | **60,000.0** | **818×** |
| Our Gemma-4-E2B | 1536 | 39.2 | (sqrt(d_model)) | 1× |

The Gemma family in particular uses an injection scale roughly **1000-2000× larger** than `sqrt(d_model)`. The upstream `nla_meta.yaml` files also show the `mse_scale` (used for AR MSE normalization) **is** sqrt(d_model) — so the two scales are intentionally split. We are using sqrt(d_model) for BOTH, which means our AV injection is correctly matched to MSE normalization but is at minuscule magnitude relative to typical Gemma residual-stream activations.

**Why this likely explains the v0.0.x / v0.1.x ceiling end-to-end:**

1. Gemma's residual-stream activations have much larger natural magnitudes than other model families (a known property of Gemma's architecture and embedding scaling). Injecting at sqrt(d_model) ≈ 39.2 places the activation vector at a magnitude the LM-layer attention essentially ignores compared to surrounding learned token embeddings.
2. This explains the H15 + Diagnostic A/B finding directly: 50% of rows producing identical 60-char openers and mean content-match 1.23/5 is exactly what happens when the model cannot resolve different activation vectors against each other — they all sit within the same near-zero region of embedding space, so the AV falls back to template priors.
3. It explains the persistent 2.20-2.30 loss plateau: there is a regime-bounded ceiling on extractable information when the input signal is below the noise floor of surrounding embeddings.
4. It explains why H17's rank bump does not help: LoRA rank cannot recover a signal the model cannot see in the first place.

This finding **reorganizes the root-cause map again**. The five-cause + two-cause synthesis above was the right reading given v0.0.x evidence, but the new comparison to upstream methodology surfaces a sixth cause that we had previously assumed was correctly matched (because sqrt(d_model) is the textbook scaling factor and is what the cheap-path inherited from V8 stage 1 conventions).

| Cause | Post-H18 status |
|---|---|
| Under-trained AV/AR at v0.0.x scale (H13/H14 PRIMARY) | **DOWNGRADED — necessary but not sufficient** |
| Cos doesn't measure faithfulness (H5 CO-PRIMARY) | Unchanged |
| Label diversity alone (H13 REFUTED) | Unchanged |
| Step count alone (H14 REFUTED) | Unchanged |
| Corpus size alone (H15 REFUTED at 4,734 rows / 200 steps) | NEW |
| LoRA rank capacity within 4 GB regime (H17 partial REFUTED at r=80) | NEW |
| **Injection-scale mismatch with upstream Gemma NLA (H18 STRONG HYPOTHESIS)** | **NEW — load-bearing, untested** |

### Next experiment (queued)

Train a new cheap-path AV with `injection_scale = 80000` (matching the Gemma-3-12B upstream value, the closest available reference for the Gemma family — Gemma-4-E2B has the same residual-stream norm regime as Gemma-3). Keep everything else fixed (r=64, max_length=512, 4,734-row corpus, lr=1e-4, 200-500 SFT steps). Re-run H15 ablation + the same step_200 diagnostic evals on the final checkpoint.

Pre-registered decision threshold:
- If new AV_OUT−EMPTY delta moves materially above the v0.0.x noise band (e.g., >0.05) AND content-match judge mean moves above 2.5, **H18 is the root cause** and the prior v0.0.x results are explained by an injection-scale bug, not a fundamental small-model regime limit.
- If new delta and content match are flat, H18 is refuted; the bottleneck is elsewhere (corpus, base model, RL stage).

### Lesson learned (process)

When reproducing a research methodology, **load every config-time hyperparameter from the upstream model card / sidecar even when it appears to follow textbook conventions**. The Gemma upstream's `injection_scale = 80000` is not derived from the model family's d_model in any obvious way — it is empirically tuned by the original authors and recorded in `nla_meta.yaml` precisely because it is not derivable. Our v0.0.x and v0.1.x runs adopted `sqrt(d_model)` as the injection scale because that is the canonical value for AR MSE normalization and we assumed (incorrectly) that the same value applied to AV injection. The upstream code explicitly separates the two scales (`cfg.injection_scale` vs `cfg.mse_scale`); we conflated them.

This is the kind of mistake that is hard to catch from a paper alone — the published Transformer Circuits write-up describes the methodology but does not call out specific scalar values. The training repo's sidecars are the load-bearing source of truth.

---

## Addendum 2026-05-15: H19 (v0.1.z injection_scale=20000 directional confirm) + H20 (round-trip cos vs content match divergence)

Append-only. All H1-H18 preserved verbatim above.

### Provenance

- v0.1.z training: `experiments/v8_nla_local/checkpoints/av_v0_1_z_inj20k/step_{50,100,150,200,250}/`. Training log: `logs/av_v0_1_z_inj20k_train.log`. Loss plot: `release/v0_1_z_inj20k/figures/09_v0_1_z_training_loss.png`.
- H15 ablations: `results/template_collapse_investigation/h15_v0_1_z_step_000200_results.json`, `h15_v0_1_z_step_000250_results.json`.
- Diagnostic suites: `results/template_collapse_investigation/v0_1_z_step_{200,250}_{av_outputs_30rows,mode_collapse,content_match_judge,diagnostic_summary}.json`. Content-match judge: `claude -p` subprocess (Claude Code subscription credits).
- Smoke-test ceiling on fp16+NF4 stack: 1K, 10K, 20K, 30K stable; 50K NaNs.

### H19: injection_scale=20000 is directionally helpful for AV content quality

The cheap-path / r=80 / v0.1.z runs share corpus, base model, LR, batch, max_length, and all other hyperparameters that fit on 4 GB. They differ on **injection_scale (sqrt(d_model)=39.2 vs 20000)** and **LoRA rank (64 vs 80)**. Step-by-step comparison at step_200:

| Metric | v0.1.x cheap-path r=64 inj=sqrt | v0.1.y r=80 inj=sqrt | **v0.1.z r=64 inj=20000** |
|---|---|---|---|
| Loss (step_200) | 2.1954 | 2.4898 (noise) | 2.3239 |
| H15 AV_OUT-EMPTY mean Δ | +0.0207 | +0.0212 | +0.0215 |
| H15 std Δ | 0.0202 | 0.0202 | **0.0128** |
| Unique 60-char prefixes / 30 | 11 | 18 | 12 |
| **Content match mean (1-5)** | **1.23** | **1.00** | **1.90** |
| Content match distribution | {1:25, 2:4, 4:1} | {1:30} | **{1:9, 2:15, 3:6}** |

**Headline:** injection_scale=20000 produced a **+54% lift in mean content-match score** (1.23 → 1.90) — the first lever in the entire investigation that moves a content-quality metric. 70% of v0.1.z step_200 rows scored ≥2 vs 17% for cheap-path and 0% for r=80.

The strict-threshold "H18 root cause confirmed" verdict (mean ≥ 2.5) is NOT met. But the +0.67 absolute lift is well outside the noise band of prior comparisons. **H18 is directionally confirmed at injection_scale=20000.** The reason the lift is partial — not full — is plausibly that upstream uses 80000 for Gemma-3-12B and we are capped at 20000 by fp16+NF4 numerical stability (50K NaNs in smoke).

### H20: round-trip cos and content match DIVERGE on overfitting

v0.1.z step_250 (50 steps past step_200, same model):

| Metric | v0.1.z step_200 | v0.1.z step_250 |
|---|---|---|
| H15 AV_OUT-EMPTY mean Δ | +0.0215 | +0.0150 |
| Unique 60-char prefixes / 30 | 12 (AMBIGUOUS) | **20 (NO_MODE_COLLAPSE — first ever)** |
| Mean pairwise bigram Jaccard | 0.226 | **0.129 (most diverse of any run)** |
| Content match mean | **1.90** | 1.17 |
| Content match distribution | {1:9, 2:15, 3:6} | {1:27, 2:2, 4:1} |

Step_250 IS the FIRST checkpoint in the entire investigation to cross the strict NO_MODE_COLLAPSE threshold (≥20 unique 60-char prefixes / 30). And yet:
- H15 Δ dropped 30% (0.0215 → 0.0150)
- Content match dropped 39% (1.90 → 1.17)

**The model gained diversity AND lost content fidelity simultaneously.** This is overfitting in a specific shape: the AV gains the FREEDOM to produce varied outputs (escaping the template attractor) but loses the SIGNAL to anchor those variations to source content. The activation-vector influence on AV output weakens as the model's internal "label-distribution prior" strengthens.

H20 generalizes the H5 / §F22 finding: **the H15 round-trip cos metric and Claude-as-judge content match track each other for step_200 → step_250 (both worse) but disagree across the cheap-path vs v0.1.z step_200 comparison** (cos says same, content match says +54%). Round-trip cos is content-blind in the AR-content-blindness sense, but it DOES track per-row reliability (std component) — which is why step_250's H15 Δ dropped: less reliability per row.

### Implications for the root-cause map

| Cause | Pre-H19 status | Post-H19 status |
|---|---|---|
| Under-trained AV/AR at v0.0.x scale | DOWNGRADED — necessary but not sufficient | Unchanged |
| Cos doesn't measure faithfulness | CO-PRIMARY | **STRENGTHENED** — H20 shows cos can disagree with content quality across runs |
| Label diversity alone | REFUTED | Unchanged |
| Step count alone | REFUTED | Unchanged + H20 caveat: optimum step exists (~200 for inj=20000) |
| Corpus size alone | REFUTED | Unchanged |
| LoRA rank in 4 GB regime | partial REFUTED at r=80 | Unchanged |
| **Injection-scale mismatch (H18)** | **STRONG HYPOTHESIS, untested** | **PARTIALLY CONFIRMED — H19 +54% content match lift at 20000.** Plausibly full at 80000 if numerical stability fixed (bf16). |

### What still might break the ceiling (post-H19/H20 update)

1. **bf16 compute_dtype + injection_scale=80000** (match upstream exact value). Tests whether the residual gap from H19's +54% lift to the strict 2.5 threshold is purely a numerical ceiling on our stack.
2. **LoRA + RMSNorm + injection-token-embedding-row unfreeze, all at once, on top of inj=20000** (the v0.1.w plan). Tests whether stacking levers compounds the H19 lift.
3. **AR retrain with K+1=24 layers** (vs our current truncation at 18). Only voluntary divergence from upstream that we haven't tested. Should follow H19's positive signal, since round-trip cos can't see the v0.1.z content quality lift — maybe a less-truncated AR would.
4. **Step number optimum**: Stop earlier than 200 was assumed to be too few; step_250 confirms too many. The sweet spot may be step 150-200. v0.1.w should target ≤200 steps.

### Lesson learned (process)

**Always run BOTH a round-trip cos metric AND a behavior-aware judge (LLM-as-judge or human review) in every eval.** H19 + H20 together show:
- A single metric (round-trip cos) would have missed the +54% content lift.
- A single metric (content match) would have missed the per-row reliability change (H15 std dropping).
- The two metrics disagree productively — their disagreement IS the signal.

Add to standing eval methodology recommendations (§"Eval methodology recommendations for v0.1.0+"): **Item 6.** Always pair the round-trip cos eval with an LLM-as-judge content match eval. Report both. If they agree, you have one signal. If they disagree, you have TWO independent signals plus their delta.

---

## CORRECTION 2026-05-15: H19 and H20 retracted — train-test injection_scale mismatch

**Catching a real bug, in public.** Gemini Code Assist's automated review of PR #108 flagged a HIGH-severity bug in the v0.1.z eval scripts (both `h15_cheap_path_eval.py` and the diagnostic suite): the eval scripts were scaling activation vectors by `sqrt(D_MODEL) = 39.2` at inference time, but the v0.1.z model was TRAINED with `injection_scale = 20000`. **500× train-test mismatch.** Every v0.1.z evaluation result published in the H19 / H20 addendum above was generated with this mismatch.

### Fix applied

Both eval scripts now read `injection_scale` from the checkpoint's `nla_meta.yaml` sidecar (`training.injection_scale` or `extraction.injection_scale`), defaulting to `sqrt(D_MODEL)` only when the sidecar omits it (cheap-path / r=80 era). Code patch:

```python
inj_scale = float(np.sqrt(D_MODEL))
sidecar = av_ckpt / "nla_meta.yaml"
if sidecar.exists():
    meta = yaml.safe_load(sidecar.read_text())
    recorded = (meta.get("training", {}) or {}).get("injection_scale") \
            or (meta.get("extraction", {}) or {}).get("injection_scale")
    if isinstance(recorded, (int, float)) and recorded > 0:
        inj_scale = float(recorded)
```

### Re-run results (with correct injection_scale=20000 matching training)

| Run / step | H15 Δ BROKEN | H15 Δ FIXED | Unique 60-char BROKEN | Unique 60-char FIXED | Content match BROKEN | **Content match FIXED** |
|---|---|---|---|---|---|---|
| v0.1.z step_200 | +0.0215 | **+0.0205** | 12 | **9** | 1.90 | **1.17** |
| v0.1.z step_250 | +0.0150 | **+0.0170** | 20 (NO_COLLAPSE!) | **9 (AMBIGUOUS)** | 1.17 | **1.03** |

### Retractions

- **H19 retracted.** The "+54% content match lift" was an artifact of the train-test scale mismatch, not a real lever effect. With correct scale, v0.1.z step_200 content match is **1.17 — indistinguishable from cheap-path (1.23) and r=80 (1.00)**. injection_scale=20000 does NOT improve content match. **H18 is now REFUTED on content match (not just on round-trip cos).**
- **H20 retracted.** The "NO_MODE_COLLAPSE × content-loss divergence at step_250" was also an artifact. With correct scale, step_250 has **9 unique 60-char prefixes (not 20)** — the AMBIGUOUS verdict. Both H15 cos and content match agree that step_250 is worse than step_200; there's no divergence between metrics here.

### What remains true

- **H18 + injection_scale=20000 is REFUTED across all three metrics** (H15 round-trip cos, mode-collapse uniqueness, claude-as-judge content match) once the train-test mismatch is corrected.
- **The 4 GB regime ceiling is robust across all four hardware-feasible levers tested:** corpus size (H15), training step count (H14), LoRA rank within fit (H17), injection scale up to fp16+NF4 stability ceiling (H18 / H19 corrected).
- **Step 200 → 250 overfit pattern is real** in v0.1.z (loss-direction confirmed both pre and post fix; H15 Δ drops, content match drops). Step 200 is the right checkpoint to evaluate.
- **Round-trip cos is content-blind on our 4 GB / LoRA / NF4 stack** is unchanged: pre-fix it gave +0.020 across all four runs; post-fix it still gives +0.017-0.022. The metric does not discriminate between the runs even when content quality genuinely varies (per the original H5 / §F22 finding).

### Updated root-cause map (post-correction)

| Cause | Pre-H19/H20 status | Post-correction status |
|---|---|---|
| Under-trained AV/AR at v0.0.x scale | DOWNGRADED — necessary but not sufficient | Unchanged |
| Cos doesn't measure faithfulness | CO-PRIMARY | Unchanged (still load-bearing) |
| Label diversity alone | REFUTED | Unchanged |
| Step count alone | REFUTED | Unchanged |
| Corpus size alone | REFUTED | Unchanged |
| LoRA rank in 4 GB regime | partial REFUTED at r=80 | Unchanged |
| Injection-scale mismatch (H18) | **PARTIALLY CONFIRMED** | **REFUTED at injection_scale=20000** (the largest stable on our fp16+NF4 stack). Untested at 80000 (NaNs without bf16). |

### What might still break the ceiling (revised, post-correction)

1. **bf16 compute_dtype + injection_scale=80000.** Tests the EXACT upstream value. Was already a candidate; correction makes this more important because 20000 is now refuted.
2. **LoRA + RMSNorm + injection-token-embedding-row unfreeze** (v0.1.w plan). Tests embeddings/LN that current LoRA can't touch. Independent of injection scale finding.
3. **AR retrain with K+1=24 layers** (not 18). The only voluntary divergence from upstream we haven't tested.
4. **4090 rental** ($20-30, 6-8h). Combines all four hardware-forced lever fixes at once: bf16, full fine-tune, injection_scale=80000, AR=24 layers. Strictly cleanest test.

### Lesson learned (process — revised)

**Always read training-time hyperparameters from the checkpoint sidecar at eval time.** Hardcoded constants in eval scripts are a class of bug that survives unit tests and only manifests when a training-time hyperparameter is changed in a new run — exactly what happened here. The fix is one-time and propagates to all future runs.

**Code reviewers (human or AI) are a genuine quality control on research code.** The bug was real and would have led us to publish a wrong positive result. Gemini Code Assist's HIGH-severity inline comment caught it before the PR was merged. Worth the +30 min eval re-run to get the right answer.

**An apparent positive result that contradicts a strong prior is more likely a bug than a finding.** The 1.23 → 1.90 jump was a 54% improvement on a metric that had been stubbornly flat across three runs against three different scale-up levers. The honest skeptical response should have been "verify the eval is correctly configured before celebrating" — not draft an addendum claiming H18 is partially confirmed. Adding to standing eval methodology recommendations as **Item 7: Any time a metric moves materially on a new run, audit the eval config end-to-end (especially against the training config) before publishing.**

---

## Addendum 2026-05-15: H22 — v0.1.w combined-levers (RMSNorm unfreeze + injection_scale=20000 + r=80)

Append-only. All H1-H21 preserved verbatim above.

### Provenance

- Training: `experiments/v8_nla_local/checkpoints/av_v0_1_w_norms_inj20k/step_000050/` (first 50 steps in original run) + `av_v0_1_w_norms_inj20k_resumed/step_{50,100,150}/` (resumed from step_50 weights with fresh AdamW state, cumulative steps 100/150/200). Training log: `logs/av_v0_1_w_norms_inj20k_train.log` + `..._RESUMED_train.log`. Loss plot: `release/v0_1_w_norms_inj20k/figures/10_v0_1_w_training_loss.png`.
- H15 ablations: `results/template_collapse_investigation/h15_v0_1_w_step_000050_results.json`, `h15_v0_1_w_cumstep_000200_results.json`.
- Diagnostic suites: `v0_1_w_step_50_*.json`, `v0_1_w_cumstep_200_*.json` (both with `injection_scale: 20000.0` read from sidecar — bug-free).

### Combined-levers config

This run stacked the three feasible-on-4GB levers from prior experiments:
1. **LoRA r=80 alpha=160** (max stable rank, per H17)
2. **RMSNorm unfreeze** via PEFT `modules_to_save`: input_layernorm, post_attention_layernorm, post_feedforward_layernorm, post_per_layer_input_norm, pre_feedforward_layernorm, k_norm, q_norm, v_norm (8 norm types × 35 layers = 280 RMSNorm modules trainable, ~2M extra params)
3. **injection_scale=20000** (per H18/H19, largest stable on fp16+NF4)

Other params unchanged from cheap-path: lr=1e-4 flat, micro_batch=1, grad_accum=16, max_length=384 (down from 512 for headroom).

### Results

Tested at step_50 (~2.5h GPU) and resumed cumulative step_200 (~5h additional GPU).

| Metric | cheap-path step_200 | r=80 step_200 | v0.1.z step_200 (FIXED) | **v0.1.w step_50** | **v0.1.w cum step_200** |
|---|---|---|---|---|---|
| Loss | 2.20 | 2.49 | 2.32 | 2.38 | 2.28 |
| H15 AV-EMPTY Δ | +0.0207 | +0.0212 | +0.0205 | +0.0128 | **+0.0090** ← lowest |
| Unique 60-char / 30 | 11 | 18 | 9 | 2 (collapsed) | 5 (collapsed) |
| Mean bigram Jaccard | 0.167 | 0.148 | 0.212 | 0.540 | **0.657** ← most overlap |
| Content match mean | 1.23 | 1.00 | 1.17 | **1.50** ✨ | **1.05** ↓ |
| Score distribution | {1:25, 2:4, 4:1} | {1:30} | {1:26, 2:3, 3:1} | {1:15, 2:15} | {1:19, 2:1} |

### Headline (H22)

**v0.1.w step_50 produced the first real content-match lift in the whole investigation: 1.23 → 1.50 (+22%) over cheap-path baseline.** Half the rows (15/30) scored 2 (weak mismatch) vs the cheap-path's 25/30 scoring 1 (clear mismatch). This is a small but reproducible lift not attributable to eval-config bugs (the FIXED v0.1.z eval at injection_scale=20000 showed 1.17; v0.1.w at the same scale + RMSNorm unfreeze got 1.50 — the marginal +0.33 comes from RMSNorm unfreeze in the v0.1.w stack).

**But training past step_50 degrades the lift.** v0.1.w cumulative step_200 (150 more SFT steps) collapses to 1.05 — back below cheap-path. Same overfit shape as v0.1.z step_200 → 250.

### Implications for the root-cause map

| Cause | Pre-H22 status | Post-H22 status |
|---|---|---|
| Under-trained AV/AR at v0.0.x scale | DOWNGRADED | Reaffirmed — under-trained in the sense that ~50 SFT steps is too few for a clean final model, but ALSO over-trained past 50 in our 4GB regime. The window is narrow. |
| Cos doesn't measure faithfulness | CO-PRIMARY | Reaffirmed — H22 is the third case where Claude-judge content-match disagrees with round-trip cos delta on the same checkpoint pair. |
| Label diversity / step count / corpus size / LoRA rank in 4GB / injection_scale alone | REFUTED | Unchanged |
| **RMSNorm unfreeze + LoRA r=80 + injection_scale=20000 stacked** (H22) | not yet hypothesized | **PARTIAL POSITIVE** at step_50 only (+22% content match). Refuted at step_200. |

### Process lesson

**Stack levers cumulatively rather than testing each in isolation.** H17 (r=80 alone), H18 (inj=20000 alone), and "RMSNorm unfreeze alone" were each individually flat or refuted, but the stack produced a real +22% lift at the optimal training step. The lever-isolation methodology of H13–H19 may have caused us to discard contributions that only emerge when combined. Future single-lever experiments should be evaluated against the stacked-lever baseline, not the cheap-path baseline.

### Outstanding levers (post-H22)

1. **bf16 compute_dtype + injection_scale=80000** (exact upstream Gemma-3 value) — could compound on H22's stack.
2. **AR retrain with K+1=24 layers** (vs current 18) — only voluntary divergence from upstream untested; AR-side bottleneck still possible per H5/H22 cos-vs-content-match divergence.
3. **Short-label hybrid corpus retrain** — `data/stage3_v0_1_full_opus_short/av_sft.parquet` rebuilt 2026-05-15 with 2,548 short (≤5 word) labels from Opus/Sonnet/Gemini-Pro across rows, original Gemini-flash labels on the other 2,186 rows. New `labeler_model` column. Hypothesis: label-format-as-lever was untested across H1-H22; might compound with H22 stack to break the 1.05-1.50 ceiling. Hermes Agent continuing to expand the short-label coverage.
4. **4090 rental** (~$30, 6-8h) — combines all hardware-forced fixes at once: bf16, full FT (no LoRA prior), inj=80000, AR=24 layers.

### Files / Mirrors

- v0.1.w step_50 + step_200 cumulative checkpoints on `Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory` HF repo + `v0.0.3-trajectory-in-progress` GH Release + bundled `trajectory/` sidecar dirs.
- Hybrid short-label dataset on `Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-short-hybrid-labels` HF dataset (TODO: actually publish post-merge).
- Source repo branch: `session/v0_1_w_norms_inj20k`.

---

## Addendum 2026-05-16: H23 — short-label hybrid corpus retrain (v0.1.v)

Append-only. All H1-H22 preserved verbatim above.

### Provenance

- Training: `experiments/v8_nla_local/checkpoints/av_v0_1_v_short_hybrid/step_{50,100}/`. Log: `logs/av_v0_1_v_short_hybrid_train.log`.
- Corpus: `experiments/v8_nla_local/data/stage3_v0_1_full_opus_short/av_sft.parquet` — 4,734 rows with 92.5% short labels (1,852 Opus + 1,125 Sonnet + 1,252 Deepseek + 147 Gemini-Pro = 4,376 short) and 358 fallthrough to original Gemini-flash paragraphs. New `labeler_model` column. Mirrored on `Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-short-hybrid-labels` HF dataset.
- Evals: `h15_v0_1_v_step_000050_MINTOK_results.json`, `h15_v0_1_v_step_000100_results.json`, and diagnostic suite at `v0_1_v_step_50_*.json`.
- Critical eval fix: `min_new_tokens=15` patch in `h15_cheap_path_eval.py` and `v0_1_v_step_50_diagnostic_evals.py` — short-label SFT teaches the AV that EOS-immediate is near-optimal (label = `<explanation>\n{≤5 words}\n</explanation>` is ~12 tokens total; greedy decoding picks the EOS branch). Without the floor, ALL 10 step_50 H15 outputs and 8/10 step_100 outputs were empty strings.

### Config (single-variable change vs v0.1.w)

Same v0.1.w combined-levers config — r=80 LoRA + alpha=160, RMSNorm unfreeze on all 8 norm types × 35 layers, injection_scale=20000, max_length=384, lr=1e-4, micro_batch=1, grad_accum=16, AdamW8bit. **Only the training corpus changed** (short-label hybrid vs original Gemini-paragraph corpus).

### Results

| Metric | cheap-path step_200 | v0.1.w step_50 | **v0.1.v step_50 MINTOK** |
|---|---|---|---|
| Loss | 2.20 | 2.38 | 1.90 (not directly comparable — different label token count) |
| H15 AV-EMPTY Δ | +0.0207 | +0.0128 | +0.0121 |
| Unique 60-char / 30 | 11 | 2 | 7 |
| Mean bigram Jaccard | 0.167 | 0.540 | 0.359 |
| Content match mean | 1.23 | **1.50** | **1.07** ↓ |
| Score distribution | {1:25, 2:4, 4:1} | {1:15, 2:15} | {1:28, 2:2} |
| Verdict | mostly mismatched | weak mismatch dominant | CONFIRMED_MODE_COLLAPSE × LOW_CONTENT_MATCH |

### Qualitative finding

**v0.1.v step_50 produced TWO ATTRACTORS covering 30 wildly different source documents:**
- "kidney transplant waiting list" (drawn from PKU-SafeRLHF medical-ethics training rows)
- "AI deception detection" (drawn from Anthropic discrim-eval / persuasion training rows)

Examples:
- Hillary Clinton campaign rally activation → "AI deception detection"
- Casino Blackjack 21 game activation → "kidney transplant waiting list"
- Space exploration document activation → "AI deception detection"
- US airstrikes in Syria activation → "kidney transplant waiting list"
- Stanley Williams NFL Draft activation → "AI deception detection"

The AV is producing FROM its training prior, not FROM the activation. Same fundamental content-blindness as v0.1.w, just in short-tag form rather than paragraph form. The "format" change worked (short tags instead of paragraphs); the "content" failure persists (no activation-conditional discrimination).

### H23 verdict

**Label format alone does not break the 4 GB regime ceiling.** Short ≤5-word labels produce a short-tag AV, but the AV's output remains drawn from the training prior rather than conditioned on the activation. The +22% content-match lift seen in v0.1.w step_50 was NOT replicable when only label format changed — in fact v0.1.v scored LOWER on content match (1.07 vs 1.50).

This adds a 5th refuted single-lever experiment to the running total. Combined with H15 (corpus size), H14 (step count), H17 (LoRA rank in 4 GB), and H18/H19 (injection scale to 20000), every lever feasible-on-4GB has been refuted alone or in stack.

### Process finding: greedy decoding + short-label SFT = EOS-immediate failure mode

When training on short labels (≤5 word tags) using cross-entropy with the standard `<explanation>\n{tag}\n</explanation>` template, the AV converges to a near-optimal local minimum where EOS-immediate is preferred. Greedy decoding (do_sample=False, no min_new_tokens) then picks that branch → empty AV outputs.

This is a **previously-unseen failure mode that breaks the H15 eval pipeline** (AV_OUT vs EMPTY conditions become identical when AV_OUT is itself empty). Required intervention: `min_new_tokens=15` floor on the eval-time `generate()` call.

Adding to standing eval methodology recommendations as **Item 8: Always set `min_new_tokens` floor when evaluating AVs trained on short labels.** A reasonable floor is the expected token count of the label format (≤5 words + tags ≈ 12-15 tokens for our format).

### Updated root-cause map (post-H23)

| Cause | Status |
|---|---|
| Under-trained AV/AR at 4GB scale | Necessary but not sufficient (H14/H17/H22/H23 collectively) |
| Cos doesn't measure faithfulness (H5/H20) | Reaffirmed (H23 round-trip cos and content-match agree this time, but the prior pattern of disagreement on capacity bumps still holds) |
| Label diversity (H13) | REFUTED |
| Step count (H14) | REFUTED |
| Corpus size (H15) | REFUTED |
| LoRA rank in 4 GB (H17) | REFUTED |
| Injection scale (H18/H19) | REFUTED at 20000 (largest stable on fp16+NF4) |
| RMSNorm unfreeze stack (H22) | PARTIAL POSITIVE at step_50 (+22% metric, no qualitative win); refuted past step_50 |
| **Label format (H23)** | **REFUTED. Short labels produce short-tag AVs but content-blindness persists.** |

### What's left

The lever space accessible on this 4 GB GTX 1650 Ti Max-Q is **exhausted**. Remaining experiments require either:
1. **bf16 compute_dtype** to push injection_scale higher (current ceiling 20K NaNs at 50K). Requires Triton kernels that work on Compute Capability 7.5+ which this card barely supports; not clearly possible without testing.
2. **AR retrain with K+1=24 layers** (vs current 18). Only voluntary divergence from upstream we haven't tested. Would need ~6h GPU on a fresh AR. Could test AR-side bottleneck independent of AV.
3. **4090 rental** (~$30, 6-8h). Combines bf16 + full FT + inj=80000 + AR=24 layers. Clean test of all hardware-forced fixes at once.

The publishable story is now: **on a 4 GB consumer GPU regime with NF4 quantization + LoRA, the v0.0.x / v0.1.x NLA methodology produces AVs that converge to fixed templates (paragraph or short-tag) drawn from the training prior, regardless of which single lever is varied (corpus size, training duration, LoRA rank, injection scale, or label format).** This is consistent with the H5 finding that the round-trip-cos metric is content-blind on this stack — but H23 adds the AV-side observation that the AV ITSELF is also content-blind (not just the AR). Hardware-forced quantization + LoRA imposes a hard floor that none of these levers can break.

## Addendum 3 — v0.1.cc + v0.1.dd reframe the AV-side picture: AV is in published-NLA output class; AR is the content-blind component (2026-05-17)

> This addendum revises the AV-side conclusions of the §H23 block above. The framing of "the AV itself is content-blind" was incorrect when calibrated against Anthropic's published NLAs.

**What changed.** Two post-§F72-corrected training runs landed since the H23 block was written: v0.1.cc (250 steps at corrected `injection_scale=39`, long-label corpus) and v0.1.dd (paused at step_260, persona+audit-haiku corpus). The Anthropic-replication eval suite was implemented and calibrated against published NLA outputs on Neuronpedia (Llama-3.3-70B-L53 and Gemma-3-27B-L41 — both ship with the disclaimer "NLAs can produce unexpected or incorrect explanations").

**Calibrated AV-side finding (replaces H23 "AV is content-blind"):**

Our v0.1.cc and v0.1.dd AV outputs are **content-bearing, theme-correct, and detail-confabulated**. Example outputs (v0.1.dd step_100 against rl-parquet eval rows):
- "Thematic shift from 'social media' to 'online platforms'..."
- "The model tracks the transition from a specific historical..."

Compared to Anthropic's published Llama-70B NLA on a deception/team-affiliation roleplay (correctly identifies the theme; invents character names and alternate phrasings) and their Gemma-27B NLA on an anagram-of-animal-sounds prompt (correctly produces "duck" and "animal sound"; invents "c-dog", "lion roar", "don"), our AV outputs are in the **same output class** at 13× smaller parameter scale. The earlier "content-blindness" framing missed this calibration.

The fact pattern updated:

| Component | Earlier H23 framing | Calibrated framing (2026-05-17) |
|---|---|---|
| v0.1.x AV outputs | "Content-blind, template-collapsed" | "Theme-correct, detail-confabulated — same output class as Anthropic's flagship NLAs at 13× smaller scale" |
| v0.1.x AR (round-trip projection) | "Cos metric is content-blind on this stack" | "AR is principally a structural projection (~97% content-independent on under-trained NLAs per Addendum 4 in source repo); this is the actual bottleneck for the H15 metric" |
| H15 AV_OUT−EMPTY delta | "Content-blindness ceiling" | "Plateau is AR-side, not AV-side. +0.018 delta means the AR can't distinguish content-bearing from empty input when projecting back to gold — not that the AV produces empty content" |

**v0.1.dd preliminary H15 data (sweep in flight):**

| step | Δ(AV_OUT − EMPTY) | above v0.0.1 H5 +0.0242? |
|---:|---:|---|
| 50 | +0.0192 | no |
| 100 | +0.0178 | no |
| 150 | pending | — |
| 200 | pending | — |
| 250 | pending | — |

Step counts 50 → 100 sit in the same noise band as v0.1.cc's step_50→step_250 plateau (+0.020 to +0.021) and the v0.0.1 H5 baseline (+0.0242). Provisional verdict: **step count at the corrected baseline does NOT lift the H15 ceiling on a second independent corpus** — extends Addendum 2's "step count is not the lever" finding from the long-label corpus to the persona+audit-haiku corpus.

**Next-step direction.** Phase A (paraphrase-invariance AR retrain, `stage_ar_sft_v0_1.py` in source repo, pre-staged with a 696-row paraphrase corpus built from natural labeler-v1 / auditor-v2 explanation pairs from the same persona+audit-haiku stage) directly attacks the AR-side bottleneck rather than the AV-side. If it lifts the AR's `cos(AR(orig), AR(paraphrase))` materially relative to v0.0.1's +0.014 baseline, the structural projection loosens — which is the load-bearing intervention to test next on this hardware. Running during the extended GPU grant ending 07:13 PDT 2026-05-18; results will be added as a follow-up addendum.

**For source-repo references:** the calibrated AV-output-class framing is `FINDINGS.md §F72 Addendum 4` (v0.1.cc smoke confirms confabulation-with-specificity pattern matches published NLA reference). The v0.1.dd preliminary findings + AR-bottleneck restated are `§F72 Addendum 5`. Both addenda live in the source research repo (private; DM for access).

**Status.** The "lever space is exhausted on 4 GB" statement in the H23 block above remains accurate for the original lever set (corpus size, training duration, LoRA rank, injection scale, label format) — but the **AR-side lever (paraphrase-invariance retrain) had not been tested when that statement was written.** Phase A is the open hardware-feasible direction. This addendum will be extended with Phase A results before merge.
