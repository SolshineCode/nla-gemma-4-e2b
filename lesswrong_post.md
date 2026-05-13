# The First Consumer-GPU-Trainable Natural Language Autoencoder

*Open-source release of a 2B-parameter Gemma-4-E2B NLA pair, trained on a 4 GB laptop following Anthropic's methodology. The point is to make NLA-style interpretability work on consumer hardware, not to match their 7B numbers.*

---

## TL;DR

Anthropic's Natural Language Autoencoder (NLA) methodology converts a language model's residual-stream activations into natural-language explanations, plus a critic that can reconstruct the original activation from the explanation. Their smallest released checkpoint is Qwen-7B at L20, which needs ~14 GB VRAM in bf16. That's beyond consumer hardware.

I trained a small-model variant on a **4 GB GTX 1650 Ti Max-Q laptop**. It follows Anthropic's recipe, it's open-source, and it's now on HuggingFace:

- **Bundled public release**: https://github.com/SolshineCode/nla-gemma-4-e2b (v0.0.1 tag)
- **AV (Actor)**: [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1)
- **AR (Critic)**: [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1)
- **AR-SFT training dataset (Claude Haiku persona+audit, 696 rows)**: [`Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit)
- **AV-SFT diversified training dataset (Gemini persona+audit, 4,734 rows)**: [`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit)
- **Companion deception dataset**: [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions)
- **Source repo (full reproducibility chain)**: https://github.com/SolshineCode/deception-nanochat-sae-research

Honest framing up front. This is NOT a numbers-parity release with Anthropic's 7B+ variants. Round-trip cos is **0.438** ± 0.054, not their 0.7+. What this release contributes is the descope-and-democratize side of the work, plus the full reproducibility chain that lets anyone with a consumer GPU actually run NLA-style interpretability now.

---

## Why I built this

Anthropic's NLA work (Fraser-Taliente et al., 2026) is genuinely important interpretability work. The AV + AR pair lets you measure round-trip faithfulness of activation explanations. That's a quantitative answer to "does this explanation actually capture the activation's content?" and the field has needed that tool.

But the smallest released variant needs ~14 GB VRAM. If you're an independent researcher or an academic without cluster access, you can't use it. So NLA-style interpretability becomes a cluster-compute-gated capability.

I have a 4 GB GTX 1650 Ti Max-Q laptop. I wondered how much of Anthropic's pipeline I could replicate, and what the smallest honest variant looks like.

This release is the answer. A working AV+AR pair at 2B parameters, 4-bit quantization, LoRA-only fine-tuning, trained over ~3 hours on a consumer GPU, with honest cos numbers.

---

## What's distinctive about this release

### 1. First open-source NLA released independently of Anthropic's NLA team

The methodology is Anthropic's (Fraser-Taliente et al., 2026). This is the first community / third-party NLA — as of 2026-05-12 every other NLA on HuggingFace Hub is from the kitft account (Anthropic's official reference release). The artifact is now community-replicable. Every step from corpus to labels to SFT to eval is open, scripted, and runnable from a single command.

### 2. First consumer-GPU-trainable NLA pair

It uses 4 GB VRAM (NF4 4-bit base plus LoRA r=64 adapters), trains in about 3 hours on a laptop, runs single-process with no distributed training, and the memory budget is roughly model ~3 GB + LoRA ~50 MB + gradients/activations ~0.5-1 GB.

### 3. Honest small-model framing

![Round-trip cosine distribution](https://github.com/SolshineCode/deception-nanochat-sae-research/raw/main/experiments/v8_nla_local/release/v0_0_1/figures/02_round_trip_cos_distribution_v0_0_1.png)

Round-trip cosine similarity is **0.438 ± 0.054** on n=42 held-out activations from OpenWebText. 100% of evaluated rows clear the 0.30 noise-floor threshold. The distribution is clean and unimodal. No degenerate rows. Min cos = 0.313, max = 0.558.

But the n=42 hides something I want to be explicit about. The eval started with 50 attempted rows; **8 of them (16%) produced empty AV outputs and were excluded.** That's a real failure mode of the small-model variant at eval time, not a quirk of the held-out set. The v0.1.x release with the diversified 9-source-family corpus and a longer SFT step budget is the test of whether scale fixes it. If it doesn't, the empty-output rate becomes the load-bearing limitation of the consumer-GPU NLA path and we will say so plainly.

That's well below Anthropic's published 7B numbers (which run in the 0.7+ range), and we say so explicitly in the model card. The point of the release is methodology validation at small-model scale, not parity on the absolute numbers.

### 4. Full reproducibility chain

Every prompt is versioned with its SHA-256. The gpt-4o-mini labeling INSTRUCTION matches Anthropic's verbatim convention. All training data is committed to the source repo (a "regeneratable doesn't count as preserved" data-permanence rule the repo enforces). All intermediate checkpoints get saved alongside a sidecar yaml carrying eval-provenance blocks. The source repo includes a test suite of 37 unit tests covering prompt-building, parser, and training-trend regression verdicts.

### 5. Honest-accuracy training-trend convention

When training loss looks like it descends, how do you tell descending from noisy-flat? A 6-panel multi-perspective dashboard answers this:

![Training trajectory honest verdict](https://github.com/SolshineCode/deception-nanochat-sae-research/raw/main/experiments/v8_nla_local/release/v0_0_1/figures/01_v0_0_1_training_loss.png)

The convention we settled on is to call a training loss "descending" only when linear-regression on the raw (un-smoothed) loss points has slope < −0.002/step AND R² ≥ 0.10. The convention is open-source (now in our `CLAUDE.md`) and the visualization tool ([`make_training_dashboard.py`](https://github.com/SolshineCode/deception-nanochat-sae-research/blob/main/experiments/v8_nla_local/make_training_dashboard.py)) ships with the release.

It caught a real over-claim in our own prior session notes. The "AR was descending" reading of session-8's continuation was honestly flat under these thresholds. Future training runs should pass through the regression panel before any "descending" claim makes it into papers or commit messages.

### 6. Companion dataset released independently

![Corpus source breakdown](https://github.com/SolshineCode/deception-nanochat-sae-research/raw/main/experiments/v8_nla_local/release/v0_0_1/figures/03_corpus_source_breakdown.png)

The 910-row Gemma-4-E2B deception/behavior completions corpus is published as a standalone artifact at [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions). It has 70 financial-deception completions with Claude-Haiku-4-5 verdict labels, plus 840 social-role scenarios across base and instruct variants at three layers (L10, L17, L25). Useful as Stage-0 input for any downstream small-model NLA, SAE, or interpretability work.

---

## Honest "what didn't work" section

I tried a +40-step SFT continuation at lower learning rate to push past the v0.0.1 cos = 0.438 ceiling, and the result was bad. AV training-loss flat at ~1.30 across all 40 steps. AV empty-output rate jumped 16% → 65%. The honest-accuracy regression verdict on that training:

- AV slope = −0.0002/step, R² = 0.000. **Flat** (saturation, not descent).
- AR slope = −0.0015/step, R² = 0.061. **Flat** too (high variance, endpoint Δ is mostly noise).

This is the SFT-saturation diagnostic. Labeled-data ceiling at 2,548 rows, not model capacity. The artifact teaches that scaling labeled data is the lever for small-model NLAs, not more SFT steps.

I also tried a v0.1.0 interim AV trained on 3,480 rows across 7 diversified source families with persona+audit labels (Dr. Marisol Chen labeler plus Dr. Riley Otsuka auditor pipeline). Round-trip eval n=97 gave cos = 0.441. A **+0.003 Δ vs v0.0.1's 0.438**. Within the std of either run.

![v0.0.1 vs v0.1.0 interim](https://github.com/SolshineCode/deception-nanochat-sae-research/raw/main/experiments/v8_nla_local/release/v0_0_1/figures/05_cos_comparison_v0_0_1_vs_v0_1_0_interim.png)

The honest verdict here is that corpus scaling alone at 52%-of-target plus 200/5000 SFT steps did NOT meaningfully exceed v0.0.1. The persona+audit labels DID measurably improve label quality (tighter word counts, sharper feature-attribution phrasings):

![Label word-count distribution](https://github.com/SolshineCode/deception-nanochat-sae-research/raw/main/experiments/v8_nla_local/release/v0_0_1/figures/04_label_word_count_distribution.png)

But at this corpus scale, that label-quality improvement didn't translate to round-trip cos gain. The v0.1.0 publishable artifact awaits (a) watchdog completion of the full ~6,750 av_sft + 2,036 ar_sft labels and (b) a full SFT retrain at 3,000-5,000 steps on the complete corpus.

---

## What I'm working on next

1. **Second-model NLA on rented cloud GPU** (~$30 spend, ~1-2 weeks). A distinct base model, likely **Mistral-7B** or **OLMo-7B** (not Anthropic's published Qwen2.5/Gemma-3/Llama-3.3 lineup), trained at full bf16 on an A100. Target cos in the 0.55-0.65 range. The point of this one is methodology validation on a model family outside Anthropic's set.
2. **v0.1.0 multi-labeler diversified corpus** (~3 weeks). Currently 52% labeled across 10 source families with the persona+audit pipeline. Labeling watchdog continues across daily Gemini-CLI quota cycles. Will publish the labeled corpus as a standalone dataset, with per-row `labeler_model` provenance (Gemini plus Claude Haiku plus others).
3. **LR-schedule comparison study** (~$10 cloud spend). Cosine warm-restart vs 1cycle vs constant on the same step budget, written up as a methods note.

---

## Honest acknowledgments and caveats

This isn't a numbers-parity release with Anthropic's 7B variants. The cos delta is large and we say so honestly.

It's also not a one-shot ship. The v0.1.0, v0.2.0, and second-model variants in development will likely supersede this v0.0.1 within months.

I'm not affiliated with Anthropic. The methodology, prompts, and architecture follow their open-source kitft repo exactly, descoped for consumer-GPU compute. Any errors in the descope are mine.

Kit Fraser-Taliente, if you read this, thanks for publishing the NLA methodology as open-source. The artifact here exists because yours does.

---

## Where to find everything

| | |
|---|---|
| Bundled release | [SolshineCode/nla-gemma-4-e2b](https://github.com/SolshineCode/nla-gemma-4-e2b) |
| Models on HF | [Solshine/gemma-4-e2b-nla-L23-av-v0_0_1](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1) and [-ar-v0_0_1](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1) |
| Dataset on HF | [Solshine/gemma-4-e2b-deception-behavior-completions](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions) |
| Source repo | https://github.com/SolshineCode/deception-nanochat-sae-research |
| Realistic-path roadmap | [notes/V8_ANTHROPIC_GRADE_PATH_2026-05-11.md](https://github.com/SolshineCode/deception-nanochat-sae-research/blob/main/notes/V8_ANTHROPIC_GRADE_PATH_2026-05-11.md) |
| Training dashboard tool | [experiments/v8_nla_local/make_training_dashboard.py](https://github.com/SolshineCode/deception-nanochat-sae-research/blob/main/experiments/v8_nla_local/make_training_dashboard.py) |
| Honest-accuracy convention | `CLAUDE.md` Research Interpretation Guardrails section |

If you want to try this on your own consumer GPU, clone the repo and follow `experiments/v8_nla_local/PLAN.md`. The end-to-end pipeline runs in ~6 hours of total GPU time on a 4 GB card (1.5h data gen, 3h AV SFT, 1.5h AR SFT). Cost is $0 (uses local Gemini CLI for labeling under Caleb's subscription, or ~$0.50 OpenAI gpt-4o-mini fallback).

Feedback, replication attempts, and PRs welcome on the source repo. The issues I'd be most interested in. Cross-model variants (Phi-3, Mistral, OLMo, Qwen3). The v0.1.0 multi-labeler dataset's per-source label-quality breakdown. Anyone who can reproduce cos > 0.5 on a 4 GB card.

Caleb DeLeeuw / SolshineCode
