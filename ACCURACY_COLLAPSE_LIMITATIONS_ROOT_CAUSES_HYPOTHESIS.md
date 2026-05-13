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
- `h3_analysis.py`, `h1_h11_eval.py`, `h2_injection_fidelity.py`, `h5_ar_gibberish.py` for reproduction
