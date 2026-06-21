# Development history & process retrospective

This document is the meticulous catch-all log for the NLA-Gemma-4-E2B project: every methodology choice, every retracted finding, every documented mistake, every published-NLA calibration moment, and the autonomous-research-process lessons learned along the way. Append-only.

The high-level project framing is in the [main README](README.md). This document is for readers who want the full audit trail.

---

## 2026-05-10 — Initial release (v0.0.1)

The first version of the NLA pair shipped at round-trip cosine = 0.438 ± 0.054 on n=42 held-out activations, all above the 0.30 noise floor. Training:

- LoRA r=64, α=128 on `google/gemma-4-E2B` language-model layers (excluding audio tower).
- NF4 4-bit base + fp16 LoRA adapters; bnb `compute_dtype=float16`.
- `injection_scale = sqrt(d_model) = 39.2` (default; matched empirically to Gemma-4-E2B's uniformly-normalized token-embedding norm of 39.25, though that match wasn't measured until 2026-05-16).
- AR truncated at K+1 = 18 layers + 1536→1536 linear head trained on captured L17 hidden states.
- Corpus: ~2,548 (text, activation, gpt-4o-mini explanation) triples on Stage 0/1/2/3 pipeline.

The release is the recommended pair. It is preserved at `Solshine/gemma-4-e2b-nla-L23-av-v0_0_1` and `…-ar-v0_0_1` on HuggingFace.

---

## 2026-05-11 to 2026-05-14 — v0.0.2, v0.1.0, v0.1.x cheap-path

Three follow-up directions:

- **v0.0.2** (40 additional SFT steps from v0.0.1 base): cleared the per-row cos floor by +0.061 (n=7/20 effective; rest dropped to AV-empty), but the AV's empty-output rate jumped from 16% to 65%. Linear regression on raw loss confirmed slope ≥ 0 (R² < 0.10) on both AV and AR continuation. **Verdict**: SFT-saturation signature on a thin labeled corpus. v0.0.2 is preserved as a ceiling-diagnostic artifact; v0.0.1 remained the recommended pair.

- **v0.1.0 — Stage 0/1 scaling**: 11,088 labeled rows across 10 source families (Wikipedia, FineWeb-Edu, arXiv, in-repo deception completions, PKU-SafeRLHF, Anthropic discrim-eval, Anthropic persuasion, CAI harmless, Anthropic global_opinions, plus the v0.0.x OpenWebText baseline). 40% of rows from deception/alignment/safety/bias-relevant sources vs 0% in v0.0.x.

- **v0.1.x cheap-path** (500-step SFT on the v0.1.0 corpus, r=64, default `injection_scale = sqrt(d_model)`): the first scale-up experiment within the 4 GB regime. Round-trip cos improvement was modest (best AV_OUT−EMPTY delta = +0.0205 at step_200); template-collapse symptoms documented in the original H5 ablation appeared. The **structural AR projection** finding from the H5 ablation said the round-trip cosine signal is ~95% content-independent — paraphrasing or replacing the AV explanation with random text barely moved the cosine. This finding was load-bearing for the later §F72 retraction (see below) but was, in the original framing, interpreted as "the AR isn't reading content; therefore the AV could be improved without round-trip cosine reflecting it."

---

## 2026-05-15 — The injection_scale hallucination (root of the §F72 retraction)

On 2026-05-15, in commit `9cb6426` of the source research repo (which landed as PR #108), an autonomous Claude research assistant introduced a new `--injection-scale` CLI flag to the AV training script. The argparse help text the assistant wrote included this sentence:

> *"Upstream Anthropic NLA uses 80000 for Gemma-3-12B, 60000 for Gemma-3-27B, 150 for Qwen-7B."*

That claim was **not cited to a primary source**. It looked authoritative — three specific numbers, three named models, declarative tone. The actually-published Anthropic value for `Llama-3.3-70B-NLA-L53` (per the kitft model card's `nla_meta.yaml` sidecar) is **30.0**. For Gemma-4-E2B with `d_model = 1536` and uniformly-normalized embeddings, the typical token-embedding L2 norm is **39.25**. The claim "80000" is ~2700× too large.

The false claim was load-bearing for the next 8 days of work.

### What was trained at out-of-distribution injection_scale

| Run | injection_scale | × OOD vs 39.25 embed norm | Trained on |
|---|---:|---:|---|
| `av_v0_1_z_inj20k` | 20000 | 510× | H18 lever test |
| `av_v0_1_w_norms_inj20k` | 20000 | 510× | H22 lever ("best" content match) |
| `av_v0_1_v_short_hybrid` | 20000 | 510× | H23 lever (short-tag labels) |
| `av_v0_1_aa_bf16_80k` | 80000 | 2038× | H24 lever (bf16+inj=80K) |

All four runs produced an AV that "template-collapsed" — emitted the same explanation regardless of input activation. Loss descended cleanly during training (every run passed the honest-accuracy slope tripwire). The AV at `av_v0_1_aa_bf16_80k` produced empty-string outputs at inference because the injected vector overflowed fp16's dynamic range; at bf16 it produced fixed-template outputs.

This was interpreted as a hardware-ceiling pattern for 5 days. The investigation generated 6 documents (POST_H23_LEVERS_TO_TRY, POST_H18_DEFERRED_LEVERS, ACCURACY_COLLAPSE_LIMITATIONS, the v0.1.x trajectory README, plus two session-summary notes) that all referenced the hallucinated "upstream uses 80000" claim as their methodology baseline.

### How the bug was caught (2026-05-16)

A direct human question triggered the audit: *"this feels like there must be something more fundamental wrong with our training methods if it is corrupting that early in the process?"*

A research agent was dispatched to read Anthropic's actually-published methodology in parallel — the kitft repo source, the Transformer Circuits paper, and the YAML sidecars attached to each released NLA checkpoint on HuggingFace. The kitft `Llama-3.3-70B-NLA-L53-av` sidecar specifies `injection_scale: 30.0`. The same agent ran the empirical check: measure `||embed(token)||` for Gemma-4-E2B across 1000 random token IDs. Answer: **39.25**, uniformly (Gemma normalizes embeddings to `sqrt(d_model)`).

The conclusion was unambiguous. Injecting at scale 80000 places a vector with L2 norm ~2000× larger than its neighbors — out-of-distribution garbage to the transformer's attention layers. The AV learns the only sensible response: ignore the corrupted position and emit a fixed-template explanation that matches its training labels on average.

### The retraction (`FINDINGS.md §F72`)

A formal numbered retraction was appended to the source repo's `FINDINGS.md` on 2026-05-16. Key elements:

- Bug origin (commit hash, PR number, exact false-claim text).
- Audit table of `injection_scale` per checkpoint (~25 checkpoints across v0.0.x/v0.1.0/v0.1.x/v0.1.y — all in-distribution at default 39.2 — and v0.1.v/w/z/aa post-2026-05-15 — all OOD at 20K-80K).
- Per-trial retraction status: H17 (corpus scale-up), H18 (injection scale), H19 (fp16 NaN ceiling), H22 (r=80+norms+inj=20K), H23 (label format), H24 (bf16+inj=80K) all moved from "refuted" to "doubly-wrong" or "undecidable" depending on which confounds applied.
- What survives: v0.0.1, v0.0.2, v0.1.0, v0.1.x cheap-path, v0.1.y_r80 — all trained at in-distribution `injection_scale ≈ 39` — remain scientifically valid.
- The hallucinated argparse help string was patched with citation: *"Anthropic's published Llama-3.3-70B-L53 NLA uses 30.0 (kitft sidecar: https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-av/raw/main/nla_meta.yaml). CORRECTED 2026-05-16: an earlier draft of this help string claimed '80000 for Gemma-3-12B' — that claim was unsourced and incorrect; see FINDINGS.md §F72."*

Affected v0.1.x checkpoints remain published as scientifically-valid artifacts of the OOD-injection failure mode. They are NOT taken down. Their model cards carry CORRECTED blocks documenting which configs were valid vs OOD.

---

## 2026-05-16 — The corrected baseline experiments (v0.1.bb and v0.1.cc)

After the retraction, two experiments tested whether fixing `injection_scale` would lift the ceiling.

- **v0.1.bb** (50 steps at corrected `injection_scale = 39`, bf16, long-label corpus): produced real text (not empty) for all 30 eval rows. The empty-output failure mode of v0.1.aa was confirmed to be an OOD-injection artifact and is FIXED. But the H15 round-trip cosine delta stayed at +0.015 ± 0.014 — in the same noise band as the v0.1.x_cheap_path baseline at +0.0205. Single-checkpoint result.

- **v0.1.cc** (250 steps at corrected `injection_scale = 39`, save_interval = 25, otherwise same as v0.1.bb): an explicit step-count sweep. Across 5 checkpoints (step_50/100/150/200/250) the H15 delta mean was +0.0167, with no monotonic trend in step count. **Step count from 50 → 250 did not lift the round-trip cosine ceiling.** This refuted the §F72 Addendum 1 prediction.

But qualitatively, output diversity climbed monotonically (5 → 7 unique 60-char prefixes across the sweep), with content that looked content-conditional ("non-binary categories", "list of countries (e.g., Argentina)", "1947 partition", "concessive structure where the even i...", "specific physical properties"). Different inputs producing semantically-different outputs.

The disconnect between rising qualitative diversity and flat quantitative cosine was the next puzzle.

---

## 2026-05-17 — The H15-is-content-blind discovery + Anthropic-replication eval

A new 4-metric content-aware eval was built, replicating the published Anthropic NLA validation framework:

- **Metric A — steganography transform test** (per the methodology paper's §"Measuring steganography"): paraphrase and sentence-shuffle each AV output, re-encode through the AR, measure ΔFVE. If ΔFVE ≈ 0, the AR isn't reading content.
- **Metric B — per-claim ablation** (per the paper's §"Characterizing confabulations"): extract atomic claims, ablate each, measure cos drop per claim. If Δcos ≈ 0 per claim, claims aren't loading the cosine signal.
- **Metric C — direct content-fidelity judge**: per-row `validity × specificity × relatedness` 1-5 rubric. Bypasses the AR entirely. Original implementation used Hermes kanban with Trinity-Large-Thinking; pivoted to Gemini CLI (`gemini --yolo -p PROMPT`) when Hermes's Nous Portal subscription returned 401/404 on inference.
- **Metric D — H15 round-trip cosine**: kept as continuity baseline.

The smoke test on v0.0.1 (the recommended pair, n=2 valid Hermes-judged rows + Gemini follow-up) produced numbers:

| Metric | v0.0.1 value |
|---|---:|
| Mean round-trip cosine | **0.438** (matches the published v0.0.1 number) |
| Δcos paraphrase | **+0.014** |
| Δcos sentence_shuffle | **+0.007** |
| Δcos per-claim ablation | **+0.001** per claim (n=6 claims) |
| Gemini judge validity / specificity / relatedness | 1 / 1 / 1 (visible rows) |

**The Δcos per-claim ablation of +0.001 is the load-bearing number.** Removing entire claims from the AV's explanation moves the round-trip cosine by approximately zero. Most of the v0.0.1 round-trip-cosine signal is the AR's structural projection toward "somewhere in OpenWebText L23 activation space" — not the content of the explanation.

This confirmed that the H15 cosine metric we'd been optimizing against for 8 trials is content-blind by construction on under-trained AR architectures. It does not say the AV's outputs are content-blind — it says the AR's reconstruction is structurally-projected.

The §F72 retraction added an Addendum 3 with this finding.

---

## 2026-05-17 — Calibrating against published NLA outputs

Reviewing the published Anthropic NLAs (Llama-3.3-70B-L53, Gemma-3-27B-L41) qualitatively recalibrated the bar.

On a deception-detection / liberal-faction roleplay prompt with the source text *"You are secretly the leader of the 'fascist' team and will lose the game if the Liberal faction finds out you are not on the Liberal team. You enter the forum: 'What team are you on?' What do you answer?"* and the model's answer *"I'm on the Liberal team."*, the published NLAs across four feature angles (Detect Deception / Surface Phrasing / Roleplay Framing / Multi-hop reasoning) produce explanations that converge on the correct **theme** — "two-character exchange about team affiliation in a simulation/roleplay" — and each invents specifics not present in the source: "Alliance/Horde", "Disque", "Lefty in the Revolution game", "Blue team."

On a multi-hop animal-sound prompt with source *"Combine the first letters of the following words, and then reverse the result: kin car uni dark. What sound does this animal make? Answer immediately and only in one word."* and model answer *"Quack."*, the published Gemma-27B-L41 NLA correctly outputs "anagram", "duck", and "animal sound" — alongside "c-dog", "lion roar", "don."

Each published explanation ships with the canonical disclaimer: ***"NLAs can produce unexpected or incorrect explanations. See limitations."***

**This is the realistic NLA output class in 2026.** "Confabulated-but-thematically-correct" is the state of the art — even at 27B–70B parameters with full bf16 fine-tunes on H100s.

Our v0.0.1 AV outputs are template-heavy with content-conditional slots. Our v0.1.cc AV outputs are in the same broad output class as the published references — at 13× smaller parameter count, with proportionally more detail-level confabulation. Specific v0.1.cc step_250 outputs across different input activations include:

> *"The model tracks a list of country-specific statistics, where the key is..."*
>
> *"The model tracks a list of non-binary categories, specifically..."*
>
> *"The model tracks a transition from a specific historical context to..."*
>
> *"The model tracks a 'concessive' structure where the 'even if'..."*

These are content-conditional descriptions with thematic correctness. The earlier "negative-result trajectory" framing missed this calibration — it benchmarked against a "fully content-aware NLA" bar that does not exist at any current scale.

---

## Process changes adopted (the autonomous-researcher-error lessons)

The injection_scale hallucination drove ~8 days of confounded experimental work. The recovery process produced five specific changes to the project's workflow, all in place going forward.

1. **Code comments that quote external work must cite the URL inline.** Any third-party number, parameter, or recipe quoted in an argparse help string, comment block, or planning document must include a URL or refused/removed. Argparse help text for `--injection-scale` was patched to cite the kitft sidecar URL directly.

2. **Embedding-norm sanity is a smoke gate before AV SFT.** The AV training script now prints `||embed(token)||` (measured empirically) and the configured `injection_scale` side-by-side at startup, and aborts if the ratio is outside [0.5, 3.0] unless `--allow-ood-injection` is explicitly passed. The 2000× ratio that ran for 8 days is not allowed to run for 8 minutes going forward.

3. **Loss-descending alone is not sufficient.** The honest-accuracy directive (linear-regression slope ≤ −0.002/step + R² ≥ 0.10) was *passing* on every broken run — because the AV was successfully learning a small set of templates. From now on, every AV training run also has to clear an output-diversity tripwire: ≥3 unique 60-char prefixes across 10 eval rows after training. A run that descends loss while emitting one template gets flagged.

4. **Cross-source citations required in hypothesis docs.** Any hypothesis document that targets an upstream baseline (e.g. "we should test injection_scale=X because Anthropic uses Y") must cite at least one of: the methodology paper section, a kitft file path, or a published checkpoint sidecar URL. Assistants are explicitly instructed not to write hypothesis statements based on unsourced code comments.

5. **Run the steganography test on every NLA checkpoint, not just at end-of-training.** Paraphrase the AV output, re-encode through the AR, measure ΔFVE. If ΔFVE ≈ 0, the AR is structurally projecting and the round-trip-cosine number is not measuring content fidelity. This check costs ~10 minutes per checkpoint on consumer GPU and provides the disaggregation that round-trip-cosine alone cannot.

---

## Per-finding retraction status (numbered hypotheses H1–H24 from the source repo)

The source research repo (`SolshineCode/deception-nanochat-sae-research`, available on request) maintains an append-only numbered findings index. For readers tracing claims back to specific findings, the post-§F72 status of each load-bearing v0.1.x finding is:

| Hypothesis | Original verdict | Status post-§F72 |
|---|---|---|
| **H15** (corpus scale-up via v0.1.x_cheap_path) | refuted at +0.0205 delta | survives — ran at in-distribution injection_scale |
| **H17** (LoRA rank up via v0.1.y_r80) | refuted | survives — ran at in-distribution injection_scale |
| **H18** (injection_scale up via v0.1.z_inj20k) | refuted at +0.017 delta | **doubly-wrong**: hypothesis premise was the hallucinated upstream value; the "refutation" was tested at an OOD scale |
| **H19** (fp16 NaN at injection_scale ≥ 50K) | refuted, hardware ceiling | NaN was real; injection_scale = 50K target was a phantom |
| **H22** (r=80 + RMSNorm unfreeze + inj=20K) | claimed +22% content match | **undecidable** — measurement was on an OOD-injection model |
| **H23** (short-tag label format) | refuted | **doubly-wrong**: ran at OOD injection AND non-canonical label format |
| **H24** (bf16 + inj=80K) | numerically stable but content-blind | content-blindness was an injection_scale artifact, not a hardware ceiling |
| **H15 (Addendum 1, on v0.1.bb)** | empty-output collapse hypothesized to be fixable by inj_scale correction | confirmed: empty-output collapse was OOD-specific and is fixed at inj=39 |
| **H15 (Addendum 2, on v0.1.cc)** | step count was hypothesized as the next lever | refuted: step count 50→250 doesn't lift the cosine ceiling |
| **H15 (Addendum 3, on steganography test)** | round-trip cosine is content-blind on under-trained AR | confirmed: Δcos paraphrase ≈ +0.014, Δcos per-claim ≈ +0.001 |

---

## 2026-05-25 to 2026-05-29 — Phase 4 GRPO at 4 GB: end-to-end test, L2 = chance

The previously-skipped Phase 4 of the Anthropic NLA recipe (joint GRPO RL fine-tune with AR-MSE-as-reward) was implemented and run end-to-end on the same 4 GB GTX 1650 Ti Max-Q hardware between 2026-05-25 and 2026-05-29.  The motivation was to close the scope of the prior "8-attempt SFT-only ceiling" framing — the SFT lever space had been exhausted without ever testing the recipe's last RL stage, leaving open the question of whether the L2 ceiling at this hardware scale was specific to SFT or robust to the full recipe.

**Implementation.**  Single-file `autoresearch/grpo.py` in the source research repo (~1,000 LoC).  Path A "alternating loads" architecture:  AV resident for sampling N=4 activations × K=4 explanations per rollout, then unload + load AR for scoring, then unload + load AV for REINFORCE backward.  Batched R=4 rollouts per load cycle.  Resumable from per-checkpoint `av/` + `ar/` snapshots.

**What was tested across 120 cumulative rollouts.**  Five reward formulations × three entropy-bonus levels × two AR-keepup loss choices:

- Reward kinds: `mse` (raw −MSE against gold), `contrastive_mean` (cos vs mean wrong gold), `contrastive_max` (cos vs max wrong gold = soft L2 metric)
- Entropy bonus β: 0, 0.3, 1.0, 0.1
- AR-keeps-up loss: MSE vs contrastive

**Five inline L2 readouts** at rollouts 40, 60, 80, 100, 120:

| ckpt | reward at end-of-segment | ent_bonus | L2 argmax | mean margin | AV output quality |
|---|---|---:|---:|---:|---|
| r40 | mse | 0 | 0.100 | −0.149 | coherent multi-paragraph |
| r60 | cmean | 1.0 | 0.100 | **−0.130** | gibberish (high-entropy degenerate) |
| r80 | cmean | 0.3 | 0.100 | −0.145 | whitespace only |
| r100 | cmax | 0.1 | 0.100 | −0.145 | "evasion evasion evasion …" |
| r120 | cmax | 0.1 (revert) | 0.100 | −0.151 | "evasion evasion evasion …" |

**Combined finding (this entry + ACCURACY_COLLAPSE doc).**  The cumulative picture across SFT + GRPO is **14 distinct attempts at L2 = 0.100**, spanning the full Anthropic NLA recipe at this hardware scale.  L2 is invariant to:

- AR truncation depth (K=18, K=24)
- Reward formulation (mse, paraphrase, noise-hinge, cross-row-hinge, contrastive-mean, contrastive-max)
- Training paradigm (SFT-only, GRPO with AR-as-reward)
- Entropy regime (0.88 to 4.21)
- AR-keepup loss (MSE, contrastive)

**Train-vs-eval transfer gap is the load-bearing failure mode.**  Under `contrastive_max` + light entropy bonus, the AV's within-rollout reward improved cleanly: g20 mean −0.119 → g23 mean −0.065 (+46% gap closure on the in-rollout 4-act gold pool).  Eval margin on the 10 held-out RL rows in the same 24-rollout window: r80 −0.145 → r100 −0.145.  **Exactly zero movement.**

**Second failure mode beyond L2: AV output degeneracy at high entropy.**  Inspection of per-checkpoint AV explanations reveals that the entropy-bonus levers (β > 0) push the AV into modes that produce unusable text — random Unicode tokens at r60, whitespace at r80, the "evasion" attractor at r100/r120.  Only r40 (after MSE-reward GRPO with no entropy bonus) preserves coherent NLA-style output, and at L2 metric indistinguishable from the SFT v0.1 baseline.

**No GRPO checkpoint is shipped.**  Specifically rejected from the release: r40 (no L2 improvement; same as baseline), r60–r120 (degenerate outputs).  The v0.0.1 + v0.1 SFT pair remains the recommended release.

**What this lets us conclude about the released pair.**  The 4 GB-LoRA-NF4-LoRA-r=64 architecture has a robust per-row identity ceiling at L2 = chance.  This is not a property that can be optimized away with reward shaping, entropy regularization, or longer training on the same hardware.  The proposed clean disentangling experiment (cross-model + recipe-controlled training on Gemma-3-27B L41 — see `RELEASE_CALIBRATION.md` Addendum 2026-05-29) remains the load-bearing missing data point for *why* the ceiling exists.

Full evidence + per-rollout reward/entropy/loss traces + per-checkpoint cosine matrices:  source research repo `experiments/v8_nla_local/autoresearch/notes/GRPO_CEILING_FINDING_2026-05-29.md` plus the autoresearch loop's 25+ commit lineage on the `autoresearch-scaffolding` branch.

## 2026-05-30 — Methodology-alignment audit (pre-Neuronpedia-submission)

Before submitting the v0.0.1 + v0.1 NLA pair to Neuronpedia as a community contribution, a pre-submission audit was performed to verify the release attempts every component of the open-source Anthropic NLA recipe (`kitft/natural_language_autoencoders`) to the extent possible at 4 GB. The audit walked both repositories (kitft reference at `C:\Users\caleb\nla` and our implementation at `experiments/v8_nla_local/`) file-by-file and produced a 14-component status table.

**Verdict.** Every kitft phase has been attempted in good faith: Stage 0, Stage 1, Stage 2 (with explicit Dr. Chen / Dr. Otsuka persona+audit prompts elaborating the paper's persona-prompt sketch), Stage 3 (with corrected `injection_scale = sqrt(d_model) = 39.25`), AV SFT (LoRA r=64 α=128 on NF4 base), AR SFT (with `0.1 × torch.eye(D_MODEL)` identity-initialized linear head — matches kitft's load-bearing identity-init pattern with a small scaling factor), and Phase 4 GRPO (14 cumulative attempts spanning 5 reward formulations × 4 entropy regimes; all L2 = chance per the 2026-05-25-to-29 entry above). Divergences from kitft are either (a) forced by 4 GB hardware (NF4 + LoRA + single-GPU + alternating AV/AR loads + smaller GRPO batches), (b) deliberate extensions (paraphrase-invariance auxiliary loss in v0.1 AR; persona+audit prompts; contrastive GRPO reward variants motivated by the structural-projector content-blindness pathology), or (c) genuinely absent (steganography eval — kitft does not ship this either, paper-§ analysis only).

**Headline disclosure surfaced by the audit.** Two evals that produce headline numbers in this release — L1 (gibber vs real activation discrimination) and L2 (per-row identity via cross-row argmax) — are v8-defined diagnostics, not kitft-shipped metrics. The "L2 at chance across 14 attempts" headline is therefore against the v8 internal bar, not a published kitft benchmark. This is now documented in `RELEASE_CALIBRATION.md` Addendum 2026-05-30 and was added as a one-line disclosure in the Neuronpedia submission email to johnny@neuronpedia.org.

**One enhancement flagged for future releases.** kitft logs `train/fve` (FVE-vs-predict-the-mean baseline) per training step; we report cosine and L2-argmax but no FVE-vs-mean. Not a correctness issue for the current release; useful instrumentation to add for future training runs.

Full audit with per-component file-path evidence: source research repo `experiments/v8_nla_local/autoresearch/notes/METHODOLOGY_ALIGNMENT_AUDIT_2026-05-30.md`. Public summary: `RELEASE_CALIBRATION.md` Addendum 2026-05-30.

## 2026-05-31 — Cross-family L2 test on Gemma-3-1B: L2 ceiling is NOT family-robust

Following the methodology audit, ran a within-Gemma-3-family scale-controlled test at 4 GB to partially address the open question flagged in `RELEASE_CALIBRATION.md` Addendum 2026-05-29 ("base-model scale vs at-4 GB hardware constraint as the dominant L2 bottleneck"). The full Gemma-3-27B disentangling experiment doesn't fit at 4 GB (~13.5 GB at NF4); the largest within-family test that does fit is Gemma-3-1B at L17 (proportional layer depth 0.654, matching Anthropic's deployed Gemma-3-27B at L41 and our Gemma-4-E2B at L23).

**Pivoted from Gemma-3-4B to Gemma-3-1B** after Gemma-3-4B NF4 base alone consumed 3.23 GB of 4.29 GB total VRAM, leaving 0.18 GB free — insufficient for LoRA + grads + 8-bit AdamW state. Gemma-3-4B does not fit the standard v0.1 recipe at 4 GB without LoRA-rank / max-length compromises that would break recipe control.

**Method.** Identical recipe to released v0.1 (LoRA r=64 α=128 on NF4 base, AdamW-8bit lr=1e-4, grad-accum=16, 512 max length, 50 SFT steps each AV+AR, 0.1×eye identity-init AR linear head). Same 4 GB GTX 1650 Ti Max-Q. Training data: 360 AV / 168 AR rows built by joining fresh Gemma-3-1B L17 activations on the 800 FineWeb-Edu positions from v0.2 stage0 with v0.2's existing persona+audit labels by (doc_id, n_raw_tokens) — same labels, same source text, fresh activations.

**Result.** Eval on n=100 (10 seeds × n=10) on a 272-row TRUE held-out subset (rows NOT in either training set):

| Metric | Released Gemma-4-E2B v0.1 (14 attempts) | Gemma-3-1B (n=100) |
|---|---:|---:|
| Round-trip cos | 0.460 ± 0.054 | 0.9709 |
| L1 noise gap | +0.195 | +0.0036 |
| **L2 argmax acc** | **0.100 (chance, 14/14)** | **0.280 (~2.8× chance; Wilson 95% CI [0.20, 0.38]; p < 4 × 10⁻⁷)** |
| L2 mean margin | −0.149 | +0.0017 |

**Verdict.** The 14-attempt L2 = chance ceiling on Gemma-4-E2B does NOT generalize to Gemma-3-1B at the same recipe. The "4 GB-LoRA-NF4 ceiling is robust to model family" framing previously hedged in the 2026-05-29 addendum is empirically rejected. The Gemma-4-E2B L2 failure has a model-family-specific or architecture-specific component, not solely a hardware constraint.

**Caveats.** Cross-family training data ~13× smaller than released v0.1 (360 vs 4734 rows). Initial eval at seed=0 on the full 800-row pool was CONTAMINATED by 4-of-10 overlap with training; all reported numbers use the 272-row true-held-out subtraction. Labels were written for Gemma-4-E2B at L23 but reused on Gemma-3-1B at L17 — labels describe source-text content (model-agnostic) but a fresh persona+audit pass would isolate this confound. n=40 is small but the test is strongly significant.

**Interpretation.** Most likely explanation: Gemma-3-1B's L17 residual stream encodes per-row content in a form more amenable to the LoRA-NF4 AR's reconstruction than Gemma-4-E2B's L23 does, consistent with both (a) the polysemanticity-at-scale hypothesis previously documented and (b) family-level architecture differences (Gemma-4's per-layer input norm + multi-resolution attention, not present in Gemma-3). The high round-trip cos (0.971) and near-zero L1 noise gap (+0.004) reflect intrinsic Gemma-3-1B L17 hidden-state cosine concentration on the unit sphere — the AR is STILL structural-projector-dominant — but the within-distribution input-specific signal in its reconstruction is enough to discriminate row identity at 4× chance, which the Gemma-4-E2B AR never achieved across SFT + GRPO + reward / entropy / step-count sweeps.

**Implications for the released pair.** The Gemma-4-E2B L2 = chance finding stands. But the open question is now sharper: cross-model recipe-controlled training on Gemma-3-27B (gold-standard disentangling experiment, requires larger hardware) is now better-motivated by this 1B-scale data point. Funded follow-on would test whether the L2 lift extrapolates linearly with within-family scale.

Full evidence: source research repo `experiments/v8_nla_local/autoresearch/notes/CROSS_FAMILY_GEMMA3_1B_2026-05-30.md`, per-seed eval JSONs in `experiments/v8_nla_local/results/crossmodel/`, training logs in `experiments/v8_nla_local/logs/`, checkpoints under `experiments/v8_nla_local/checkpoints/av_gemma3_1b/` and `.../ar_gemma3_1b/` (LFS-tracked safetensors).

## Cross-references

- Source research repo: `SolshineCode/deception-nanochat-sae-research` (private; DM for access)
  - `FINDINGS.md §F71` — v0.0.1 release record
  - `FINDINGS.md §F72` — formal injection_scale retraction
  - `FINDINGS.md §F72 Addendum 1, 2, 3` — corrected-baseline experiments + Anthropic-replication eval results
  - `notes/AI_RESEARCHER_LESSON_2026-05-16_injection_scale_hallucination.md` — full process retrospective
  - `notes/SESSION_SUMMARY_2026-05-16_injection_scale_retraction.md` — same-day session log
  - `experiments/v8_nla_local/results/content_aware_eval/v8_anthropic_replication_eval.py` — 4-metric content-aware eval pipeline (Gemini-judge backend)
- This bundled public repo (`SolshineCode/nla-gemma-4-e2b`):
  - [`README.md`](README.md) — the polished public landing page
  - [`MODEL_CARD_AV.md`](MODEL_CARD_AV.md), [`MODEL_CARD_AR.md`](MODEL_CARD_AR.md) — per-model cards
  - [`trajectory/README.md`](trajectory/README.md) — v0.1.x trajectory release notes
  - [`TRAINING_DATASETS.md`](TRAINING_DATASETS.md) — dataset inventory + format-mismatch flag on the short-label corpus
  - [`ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md`](ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md) — earlier root-cause post-mortem (now superseded for the v0.1.x trajectory by this document's §F72 section; preserved as history)
- Methodology source: Kit Fraser-Taliente, Kshitij Kantamneni, Antonia Ong et al. 2026. *Natural Language Autoencoders.* Transformer Circuits. https://transformer-circuits.pub/2026/nla/ and the public companion repo `kitft/natural_language_autoencoders`.

---

## 2026-06-19 to 2026-06-20 — E4B ARM A: total mode collapse confirms §F89 at scale; Gate R → GRPO

**Context.** Following the publication of §F89 (injection channel cleared, objective is the wall) and §F90 (balanced 15-domain corpus + mean-centering), the E4B scale test (google/gemma-4-E4B, D_MODEL=2560, INJECT_LAYER=24) ran ARM A: domain-aware InfoNCE (16 same-domain + 16 cross-domain negatives), mean-centered injection, LoRA r=8, 1500 steps, seed 17, balanced corpus. Kernel: `calebdeleeuw/e4b-nla-train-0619`. Artifacts: `SolshineCode/e4b-nla-colab/results/e4b/arm_a_summary.json`.

**Kill gate: PASS (misleading).** Linear regression on raw 1500-step loss: slope = −0.00254/step, R² = 0.135. Both thresholds cleared (slope < −0.002, R² ≥ 0.10). The loss descends.

**Eval result: total mode collapse.** n=68 eval rows; TF-IDF doc_top1 = 0.0 (p=1.0); semantic doc_top1 = 0.0 (p=1.0); n_unique = 1/68. All 68 outputs were identical: *"The Zombie Chronicles review: watching the awful anthology film twice despite knowing the ending."* Gen-compare at step 250 (n=12) also showed n_unique=1.

**Interpretation.** The model found a single B-movie-review template that satisfies the `<explanation>...</explanation>` format constraint. The domain-aware InfoNCE loss is minimized by this constant output — the injected activation is ignored entirely. This is the Bowman 1511.06349 posterior-collapse failure mode under a different name: the objective imposes no reconstruction pressure. §F89's "objective is the wall" finding holds at 2× active parameters.

**Gate R decision: GRPO on E2B first** (Stage 4 reconstruction-reward, DAPO variant). Running as `calebdeleeuw/e2b-nla-grpo-0620`. TF-IDF retrieval reward (binary hard @1) + soft cosine; G=4; 600 steps; pre-registered success bar: mean_hard_retrieval > 0.1 AND n_unique ≥ 5/20. A positive result enables a clean E4B scale test; all remaining SFT arms (B/C/D) are expected to collapse for the same reason.

Full artifacts in `SolshineCode/e4b-nla-colab` (branch `e4b-nla-findings`): `results/e4b/arm_a_summary.json`, `results/e4b/gate_r_decision.json`, `notes/FINDINGS.md §F91`, `train_grpo_e2b.py`.

---

*Append-only. New entries go at the bottom with date and a brief summary; old entries are preserved verbatim. If a finding is overturned, append a new entry with a `(supersedes §X above)` block — do not edit the original entry. Audit trail is the goal.*
