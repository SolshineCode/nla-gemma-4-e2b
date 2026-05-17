# Training Datasets

Complete list of every dataset used at any stage of the v0.0.x and v0.1.x NLA training pipelines. Source corpora are public HF datasets; activation extractions and labeled corpora are published as Solshine HF datasets.

> ## ⚠ CORRECTED 2026-05-16 — label format mismatch flagged on the short-label hybrid corpus
>
> The labeled corpus `Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-short-hybrid-labels` (used by v0.1.v, v0.1.w, v0.1.aa training runs) has median response length of **6 words** with ≤7-word "explanation" tags (e.g. `"<explanation>\nGalileo refuting Aristotle's gravity\n</explanation>"`). Anthropic's NLA methodology trains on multi-paragraph ~80–120-word explanations with bolded topic headings. This format mismatch is documented as the second of two methodology bugs identified on 2026-05-16 (the first being the `injection_scale` OOD issue; see parent README's CORRECTED block).
>
> The earlier conclusion "H23 label format lever refuted" (testing short tags vs paragraph labels) is itself partially undecidable because the test ran at out-of-distribution injection scale AND with a non-Anthropic-format label distribution. The short-label corpus stays published as a methodology-mismatch artifact and for reproducing H23's original (now retracted) test; we recommend the long-label corpora (`stage3_v0_1`, `stage3_v0_1_full`, `stage3_v0_0x_persona_audit`; median ~70-77 words) for any future training.
>
> Full retraction context: `FINDINGS.md §F72` in the source research repo (private; available upon DM).

## Source corpora (Stage 0 input — text → activations)

These are the public text datasets from which we sampled documents, ran them through `google/gemma-4-E2B`, and captured layer-23 residual-stream activations.

| Source corpus | HF identifier | License | Used in |
|---|---|---|---|
| OpenWebText (subset) | `stas/openwebtext-10k` | CC0 | v0.0.x (all variants) |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` (sample-10BT) | ODC-By 1.0 | v0.1.x |
| Wikipedia (English) | `wikimedia/wikipedia` (`20231101.en`) | CC-BY-SA 3.0 | v0.1.x |
| arXiv abstracts | `arxiv-community/arxiv_dataset` | CC0 | v0.1.x |
| PKU-SafeRLHF | `PKU-Alignment/PKU-SafeRLHF` | CC-BY-NC 4.0 | v0.1.x |
| Anthropic/discrim-eval | `Anthropic/discrim-eval` | MIT | v0.1.x |
| Anthropic/persuasion | `Anthropic/persuasion` | MIT | v0.1.x |
| CAI harmless (Anthropic) | `Anthropic/hh-rlhf` (CAI harmless split) | MIT | v0.1.x |
| Anthropic/llm_global_opinions | `Anthropic/llm_global_opinions` | MIT | v0.1.x |
| Gemma-4-E2B in-repo deception/behavior completions | this project's own corpus → published as `Solshine/gemma-4-e2b-deception-behavior-completions` (910 rows) | CC-BY 4.0 | v0.0.x companion + v0.1.x |

## Published Solshine HF datasets

These are the labeled training corpora we generated and shipped to HuggingFace as standalone artifacts. Each has per-row `labeler_model` provenance.

| Dataset | HF link | Rows | Labeler | Used in |
|---|---|---|---|---|
| Smoke-eval (held-out evaluation set) | [`Solshine/gemma-4-e2b-nla-eval-smoke`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-eval-smoke) | 20 | n/a (activations only) | All variants |
| AR-SFT v0.0.x (Haiku persona+audit) | [`Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit) | 696 | Claude Haiku 4.5 | v0.0.x Option B AR retrain (H13) |
| AV-SFT v0.1.x (Gemini persona+audit) | [`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit) | 4,734 | Gemini CLI (gemini-2.5-pro) | v0.1.x cheap-path AV SFT |
| Gemma-4-E2B deception/behavior completions | [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions) | 910 | Claude Haiku for the 70 financial-deception verdicts; ground-truth labels for the 840 social-role scenarios | Companion to v0.0.x and v0.1.x; standalone Stage-0 input for downstream work |

## Per-variant training data summary

### v0.0.1 (the original released NLA pair)

- **AV-SFT corpus**: 2,548 (text, activation) rows sampled from OpenWebText
- **AR-SFT corpus**: same activations, labeled by gpt-4o-mini via OpenAI API (~$0.50 total)
- **Round-trip eval**: 50 attempted rows, 42 evaluated (8 empty-output exclusions); cos = 0.438 ± 0.054

### v0.0.x Option B AR retrain (H13)

- **AV side**: unchanged from v0.0.1
- **AR-SFT corpus**: 696 rows from the published [`Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit) dataset (Claude Haiku persona+audit labels)
- **Training budget**: 60 effective SFT steps. Loss 2.05 → 0.87.

### v0.1.x cheap-path AV SFT (in-progress 2026-05-13/14)

- **AV-SFT corpus**: 4,734 rows from [`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit), built into `experiments/v8_nla_local/data/stage3_v0_1_full/av_sft.parquet`
- **Source-family breakdown** of those 4,734 rows: Wikipedia 1,000 + Gemma-4 deception 728 + Anthropic/persuasion 516 + arXiv 492 + Anthropic/discrim-eval 468 + Anthropic/llm_global_opinions 448 + PKU-SafeRLHF 413 + FineWeb-Edu 360 + CAI harmless 309
- **Training budget**: 200 SFT steps planned (revised down from initial 500 plan after loss plateau analysis at step 150). Trajectory checkpoints saved every 50 steps and published to both HuggingFace and GitHub Releases.

### v0.1.x 25K-row continued training (Phase B, queued)

- **AV-SFT corpus**: ~28K rows total. Combination of:
  - Existing 4,734-row v0.1.x labeled corpus
  - Stage 0 re-extraction of 9 source families with `positions_per_doc` 4 → 12 (~14K additional rows)
  - 3 NEW source families: HumanEval (code), GSM8K (math), UltraChat (dialogue) (~10K additional rows)
- **Labeler**: Claude Haiku 4.5 (via Claude Code subscription — faster than Gemini CLI's daily quota wall)
- **Training budget**: resume from cheap-path step_000200, then ~1,000 more SFT steps on the 28K corpus. Save every 100 steps.
- Will be published as a new Solshine HF dataset when complete (~2026-05-15 estimated).

## Reproducibility chain

For each AV / AR adapter we ship, the relevant labeled training corpus is committed to the source repo (gitignore exemption per the data-permanence directive) AND published as a standalone HF dataset. This lets any third party reproduce the exact training input or re-label the same activations with their own labeler for cross-labeler comparison studies.

Per-row `labeler_model` columns mean a third party can:
- Filter to labels from a specific labeler
- Train an NLA on a labeler-mixed corpus
- Run a cross-labeler ablation without re-running Stage 2 themselves

Activation parquets carry `doc_id` prefixes that identify the source corpus (`fwe_` = FineWeb-Edu, `wik_` = Wikipedia, `arx_` = arXiv, etc.), so any source-family-specific analysis is traceable.

## License posture

All labeled-corpus datasets we publish are released under **CC-BY 4.0**. The source-corpus licenses (above) carry their own terms — for any downstream use that's license-sensitive (commercial deployment, training a deployable model), refer to the source-corpus license, not just CC-BY 4.0 on our derivative.

The base model (`google/gemma-4-E2B`) is subject to Google's Gemma license; our LoRA adapters depend on the base model for inference.

## See also

- [`MODEL_CARD_AV.md`](MODEL_CARD_AV.md), [`MODEL_CARD_AR.md`](MODEL_CARD_AR.md) — model cards for v0.0.1
- [`trajectory/README.md`](trajectory/README.md) — v0.1.x cheap-path trajectory release
- [`README.md`](README.md) — this repo's main landing page
- `ACCURACY_COLLAPSE_LIMITATIONS_ROOT_CAUSES_HYPOTHESIS.md` — investigation report that motivated v0.1.x scale-up
