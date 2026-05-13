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

*Round-trip cosine similarity distribution for the v0.0.1 NLA pair on 42 held-out activations. Clean unimodal distribution centered at 0.438 ± 0.054. 100% of evaluated rows clear the 0.30 noise-floor threshold. No degenerate (near-zero) rows. Min 0.313, max 0.558. The matching n=50 attempted minus 8 empty-output exclusions explains the 42 evaluated. See "Limitations" for the per-row diversity caveat that this single number does not capture.*

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

### ⚠ Read this before using v0.0.1 for interpretability work

The cos number above is **principally a structural artifact**, not a measurement of explanation faithfulness. A 5-experiment investigation against v0.0.1 (documented in full in `ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md`) established three load-bearing findings:

**1. The AR is content-blind.** Mean cos by AR input condition on n=10 rows (same target activations as the published eval):

| AR input | Mean cos | Above 0.30 floor |
|---|---|---|
| Real AV-generated explanation | 0.4292 | 10/10 |
| Random unrelated Wikipedia sentences | 0.4045 | 10/10 |
| Random nonsense tokens (`"qwop fnar blarp..."`) | 0.4135 | 10/10 |
| **Empty string** | **0.4051** | **10/10** |

The "explanation" contributes a mean **+0.024 cos** over feeding the AR nothing at all. About 95% of the published cos number comes from the AR's content-independent projection from any text input toward OpenWebText activation space; about 5% from explanation conditioning. **All four conditions clear the 0.30 noise floor 10/10 times.**

**2. The AV does not condition outputs on the activation's semantic content.** Under greedy decoding, the same 4 opening template stems (`legal case` / `protest` / `new feature` / `new policy`) are emitted regardless of source. Under sampling (temperature 0.7), outputs are 100% lexically unique but **topically wrong**: a Hillary Clinton campaign rally activation produces "future of work" / "merger" / "police investigation" depending on sampling seed. Random gaussian unit vectors injected in place of real activations produce more template variation (7/10 unique) than real activations (6/10). Activation direction matters for empty-vs-content behavior but the mapping to content is essentially arbitrary.

**3. The originally-reported "4 templates / 298-char outputs" finding was partly a storage bug.** `eval_round_trip.py:195` stored `"explanation": explanation[:300]` in the result JSON; the actual model emits 520-605 chars per row under greedy. The prefix is templated, but the suffix variation was hidden by the truncation. The bug is fixed in this release; full per-row outputs from the rerun ship in `experiments/v8_nla_local/results/template_collapse_investigation/` in the source repo.

**Why the failure happened:** the AV saw approximately 12% of its training data once (220 effective rows out of 1,852 at 55 SFT steps × grad-accum 4 × batch 1). The AR was similarly under-trained. The model converged to "produce the trained NLA output form" without learning to condition body content on the injected vector or to use explanation text on the AR side. This is the predicted small-model failure mode under aggressive descope.

The v0.1.x interim AV (trained on diversified persona+audit labels at 200 SFT steps) at equivalent cos (0.441) showed 55 unique full-output patterns vs v0.0.1's 5. Label diversity and training scale are the load-bearing levers per the investigation; v0.1.0 full corpus + 3,000-5,000 SFT steps is the next experiment.

## What this artifact is and is not

**v0.0.1 is most useful for:**

- ✅ **Methodology replication** — the full pipeline runs end-to-end on a 4 GB consumer GPU
- ✅ **Baseline for v0.1.x scaling experiments** — already informative (4 templates v0.0.1 vs 55 templates v0.1.x interim at equivalent cos)
- ✅ **Infrastructure starting point** — multi-labeler pipeline, persona+audit prompts, restart-safe chunked output, honest-accuracy training-trend convention are all independently reusable
- ✅ **The published HF datasets** are useful as Stage-0 inputs to anyone else's NLA, SAE, or interpretability work
- ✅ **The descope choices** (NF4+LoRA, forward-hook injection, AR truncation) are documented and reusable

**v0.0.1 is NOT yet useful for:**

- ❌ **Per-row activation interpretation** — "what does THIS activation mean?" gets you one of 4 templates, not a faithful description
- ❌ **Cross-activation comparison** — bucketing 4 ways carries near-zero information for differentiating activations
- ❌ **Downstream classifier feature** — 4-bucket label space gives a classifier almost no signal
- ❌ **A faithfulness certificate for some other NLA you're evaluating** — cos and per-row faithfulness are dissociable per our finding; cos alone is not sufficient

For per-row faithful explanations, wait for v0.1.0 (in progress, ~3-6 weeks ETA) or the second-model 7B variant. v0.0.1 is the smallest honest variant of the methodology, not a deployable interpretability tool.

---

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

## How v0.0.1 fits the constraints (the tricks)

### Hardware (fit Gemma-4-E2B + training on 4 GB VRAM)

- **NF4 4-bit base weights.** Cuts the ~4 GB bf16 model to ~1 GB.
- **LoRA r=64 instead of full fine-tuning.** Only ~50 MB trainable, not billions of params.
- **bf16 LoRA on top of NF4 base.** Mixed precision; gradients fit.
- **Gradient checkpointing.** Recomputes activations on backward; trades ~30% compute for ~40% VRAM.
- **AdamW 8-bit optimizer.** Vs 32-bit AdamW, which holds 8 bytes/param of optimizer state and would not fit.
- **`micro_batch=1` + `grad_accum=16`.** Effective batch 16 without the VRAM cost of a real batch 16.
- **`max_length=512` context.** Vs 2048+ standard. Cuts activation tensors 4×.
- **AR truncation: first 18 of 35 layers + Linear(1536, 1536) head.** Forward pass only through half the model.
- **Forward-hook injection on embedding layer (NOT `inputs_embeds`).** Gemma-4 OOMs on `inputs_embeds`; the in-place hook variant fits 4 GB.
- **`captured[0] = ...` in-place overwrite, not `captured.append(...)`.** Fixed a 240 MB residual leak that would have spilled to CPU on long generations.

### Time (3 weeks of evenings, ~6 GPU-hours per full run)

- **Restart-safe chunked parquet output.** Per-batch chunks in a `chunks/` subdir; relaunches skip already-done batches. Saved 2+ days when Gemini quota walls hit mid-run.
- **Watchdog auto-resume across Gemini daily quota cycles.** Continues labeling across the 24h reset wall without manual relaunch.
- **`PYTHONUNBUFFERED=1` / `python -u`** on every long training run. Avoided a 2-GPU-hour false "stuck" diagnosis from block-buffered stdout.
- **`save_interval=50`** (not 100 or 500). First checkpoint lands inside ~2.5 hours so we get fast confirmation training is actually working.
- **Squash-merge PR cadence.** Clean main history, parallel feature branches without merge hell.
- **Sidecar `nla_meta.yaml` per checkpoint.** Eval-provenance lookup in 1 file open instead of grepping result JSONs.

### Budget ($0 cloud, ~$0.50 in API spend total)

- **Local 4 GB GPU.** Zero cloud compute spend for v0.0.1 training and eval.
- **Gemini CLI in YOLO mode under personal Gemini Pro subscription.** Free labeling for the v0.1.x diversified corpus.
- **Claude Code credits for Claude Haiku labeler.** Already paid for; zero marginal cost on the 696-row v0.0.x ar_sft re-label.
- **gpt-4o-mini fallback only when needed.** v0.0.1's full original labeling was ~$0.50 total.
- **Synthetic personas (Dr Chen + Dr Otsuka) instead of real LLM judges.** Two cheap LLM calls per row, no judge-API surcharge.
- **HuggingFace free tier for hosting.** Model repos + dataset repos at $0/mo.
- **GitHub free tier for public repo storage.** Including LFS-free parquet commits under the ~50 MB threshold.
- **Free public corpora.** OpenWebText, FineWeb-Edu, Wikipedia, Anthropic safety datasets, PKU-SafeRLHF, arXiv abstracts.
- **Round-trip eval done locally.** No API charges for the cos measurement.

### Methodology (descope stays faithful, not corner-cutting)

- **Persona+audit labeling pipeline.** Dr Chen labeler then Dr Otsuka auditor instead of one-shot prompts. Tighter labels, two passes preserved per row.
- **Per-row `labeler_model` provenance column.** Future cross-labeler ablations happen without rerunning anything.
- **Honest-accuracy training-trend verdict (slope < −0.002/step AND R² ≥ 0.10).** Caught a real over-claim before it shipped.
- **Data-permanence directive.** Commit every parquet to `results/` immediately; "regeneratable from script" doesn't count.
- **Eval-provenance sidecar convention.** Commit SHA + parquet SHA-256 + headline cos numbers in YAML alongside each checkpoint.

### Software / tooling (Windows-specific gotchas)

- **`shutil.which("gemini")`** to resolve npm `.CMD` shims. `subprocess.run(["gemini", ...])` can't find them; `subprocess.run([shutil.which("gemini"), ...])` does.
- **`MSYS_NO_PATHCONV=1 taskkill /F /T /PID`** to kill stuck Python processes from Git Bash without MSYS rewriting `/F` to a Windows path.
- **`device_map={"": torch.cuda.current_device()}` (integer)** not `{"": "cuda"}` (string). The string form silently falls back to CPU on bitsandbytes 0.49.2 with no error.
- **`KMP_DUPLICATE_LIB_OK=TRUE`** prefix on every run to dodge the torch+numpy OpenMP conflict on Windows.

**Total spend for v0.0.1:** ~$0.50 in API charges + $0 cloud + electricity for ~6 GPU-hours on a laptop. Time: 3 weeks of evenings, with the failed continuation experiment and the AR-descending false alarm both folded in.

## Training trajectory (honest)

![Training loss](figures/01_v0_0_1_training_loss.png)

*Six-panel honest-accuracy training dashboard for the v0.0.1 AV+AR continuation experiment. Each panel shows the same raw loss points with a different smoothing or regression overlay. The bottom-right panel reports the linear-regression slope and R² used to adjudicate the "descending vs flat" verdict (threshold: slope < −0.002/step AND R² ≥ 0.10). This is the tool that caught a real over-claim in the project's own session-8 notes (AR "descending" was honestly flat under these thresholds). Regenerate with `python experiments/v8_nla_local/make_training_dashboard.py --auto` from the source repo.*

The continuation experiment shows the **SFT-saturation diagnostic**. AV training-loss flat at ~1.30 across +40 steps. AV's empty-output rate jumped from 16% to 65% in that window. This is the limit of useful single-epoch SFT on a 2,548-row corpus, not a model-capacity ceiling. The result motivates the **v0.1.0 scaling work** (in progress, corpus growing to ~11K rows across 10 diversified source families).

**Honest-accuracy verdict on the +40 continuation** (linear-regression on raw loss):

- AV. Slope = −0.0002/step, R² = 0.000. **Flat** (saturation, not descent).
- AR. Slope = −0.0015/step, R² = 0.061. **Flat** too. High variance, endpoint Δ is mostly noise.

The verdict thresholds we use (per the convention added to `CLAUDE.md`) call a training loss "descending" only when slope < −0.002 AND R² ≥ 0.10. This convention is itself a community contribution. We publish a 6-panel diagnostic dashboard tool alongside the model: `make_training_dashboard.py` in the source research repo (currently private, **available upon request — DM me**).

It caught a real over-claim in our own prior session notes. The "AR was descending" interpretation of session-8's continuation was honestly flat under these thresholds.

## Companion datasets on this release

![Corpus source breakdown](figures/03_corpus_source_breakdown.png)

*Source-family breakdown of the v0.1.x diversified training corpus (4,734 rows). The 9-family mix is deliberate: alongside web text (FineWeb-Edu, Wikipedia, arXiv), the corpus reaches into alignment-relevant text (Anthropic safety datasets, in-repo Gemma-4-E2B deception completions, PKU-SafeRLHF) so a downstream NLA can pick up deception-relevant cues, not just generic OpenWebText semantics.*

![Per-row round-trip scatter](figures/06_per_row_round_trip_scatter.png)

*Per-row round-trip cos for the 42 evaluated rows (v0.0.1, n=50 attempted, 8 empty-output exclusions). The horizontal line at 0.30 is the noise floor; 100% of evaluated rows clear it. Min 0.313, max 0.558.*

![v0.0.1 vs v0.1.0 interim](figures/05_cos_comparison_v0_0_1_vs_v0_1_0_interim.png)

*Headline cos comparison. v0.0.1 (n=42) at 0.438 ± 0.054 vs v0.1.x interim AV (n=97) at 0.441 ± 0.052. The interim was trained on the diversified persona+audit corpus at only 200 SFT steps and 52% of the eventual v0.1.0 target corpus; corpus scaling alone at that step count produced +0.003 cos (within either run's std). This is what motivated the "label diversity is the load-bearing variable, not SFT steps alone" reading documented in Limitations below.*

![Label word-count distribution](figures/04_label_word_count_distribution.png)

*Label word-count distribution for the v0.1.x persona+audit corpus vs the v0.0.x baseline gpt-4o-mini labels. The persona+audit pipeline (Dr Chen labeler + Dr Otsuka auditor) produces tighter, more uniform label lengths. The label-quality improvement did not translate to round-trip cos gain at this corpus scale, but the underlying label artifact is independently useful for cross-labeler comparison studies.*

### Available HF datasets

This v0.0.1 release ships alongside three published HuggingFace datasets:

- [`Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit) — 696-row AR-SFT training corpus, Claude Haiku persona+audit
- [`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit) — 4,734-row AV-SFT diversified training corpus, Gemini persona+audit, 9 source families
- [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions) — 910-row companion deception/behavior corpus. 70 financial-deception completions with Claude-Haiku-4-5 verdict labels plus 840 social-role scenarios across base and instruct variants at three layers (L10, L17, L25). Useful as Stage-0 input for any downstream small-model NLA, SAE, or interpretability work.

## Intended use

- Interpretability research on Gemma-4-E2B without requiring cluster compute
- Stage-0 input for sparse-autoencoder feature discovery on the same base model
- NLA-methodology validation in the consumer-GPU regime
- Baseline checkpoint for future small-model NLA work (e.g., Gemma-3-4B, Phi-3, Mistral-7B variants in development)
- Activation-explanation tool in deception-research or safety-monitoring pipelines (paired with a downstream classifier)

## Limitations and honest framing

- **Cos = 0.438 is far below Anthropic's published 7B numbers (~0.7+).** Use this artifact for methodology replication, not for matching their absolute performance.
- **Per-row explanation diversity is low at this SFT scale (template collapse).** Concrete: 20 unique full-explanation strings across 42 evaluated rows, 52% exact-duplicate rate, 4 opening stems at 80-char granularity, 81% of rows return the same "legal case" template stem. Round-trip cos and per-row faithfulness are dissociable: cos clears the noise floor even when the AV labels a Hillary Clinton campaign rally activation with the "legal case" template stem. Full root-cause analysis (5 contributing causes including under-trained AV, low-diversity labels, no SFT diversity penalty, joint-pair-cos eval blindness, and OpenWebText homogeneity) lives in `ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md` in the source repo. The v0.1.x interim AV at equivalent cos (0.441) shows 55 unique patterns across 97 rows, supporting the hypothesis that label diversity (not SFT step count alone) is the load-bearing lever.
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
# For a complete, self-contained inference example that takes your own text,
# extracts an L23 activation, runs the AV+AR round-trip, and prints cos,
# see `examples/round_trip_example.py` in the public bundled release:
# https://github.com/SolshineCode/nla-gemma-4-e2b
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

- Source research repo (full reproducibility chain). `SolshineCode/deception-nanochat-sae-research` — currently private, **available upon request — DM me**.
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
