# NLA-Gemma-4-E2B

**First open-source Natural Language Autoencoder (NLA) released independently of Anthropic's NLA team.** Trained end-to-end on a 4 GB consumer GPU. The methodology contribution at small-model scale to democratize NLA research.

<img width="2752" height="1536" alt="Gemini_Generated_Image_gqbm6agqbm6agqbm" src="https://github.com/user-attachments/assets/7c2dbb86-03a2-4fb5-b234-b7645175825a" />

This is the **bundled public release** of the Gemma-4-E2B NLA pair (v0.0.1) plus its labeled training corpora. The full reproducibility chain (corpus extraction, persona+audit labeling, AV+AR SFT, round-trip eval) is in the [source research repo](https://github.com/SolshineCode/deception-nanochat-sae-research).

---

## What's distinctive about this release

- **First open-source NLA from outside Anthropic's NLA team.** As of 2026-05-12, every other NLA on HuggingFace Hub is under the `kitft` account (Kit Fraser-Taliente, the paper's first author and Anthropic's official reference release). The methodology is Anthropic's. The artifact here is the first third-party reproduction.
- **First consumer-GPU-trainable NLA pair.** End-to-end on a 4 GB GTX 1650 Ti Max-Q laptop. About 3 GPU-hours total.
- **Honest small-model framing.** Round-trip cos = 0.438 ± 0.054 on n=42 held-out activations. 100% above the 0.30 noise-floor threshold. NOT a numbers-parity release with Anthropic's 7B variants (cos 0.7+).
- **Full reproducibility chain.** Corpus to labels to SFT to eval is single-command runnable. Every prompt versioned with SHA-256.
- **Multi-labeler training corpus.** Per-row `labeler_model` provenance across Claude Haiku and Gemini CLI labelers, with a persona+audit pipeline (Dr. Marisol Chen labeler + Dr. Riley Otsuka auditor) replacing the bare gpt-4o-mini convention.
- **Honest-accuracy training-trend convention.** A 6-panel dashboard tool plus regression-based "descending vs flat" thresholds (slope < −0.002/step AND R² ≥ 0.10).

---

## Where everything lives

### HuggingFace artifacts (4 repos)

**Models:**
- [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1) — AV (Actor) adapter, NF4 + LoRA r=64
- [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1) — AR (Critic) adapter + linear head

**Datasets:**
- [`Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-ar_sft-v0_0_x-haiku-persona-audit) — 696-row AR-SFT corpus, Claude Haiku persona+audit labels
- [`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit) — 4,734-row AV-SFT corpus, Gemini persona+audit, 9 source families
- [`Solshine/gemma-4-e2b-deception-behavior-completions`](https://huggingface.co/datasets/Solshine/gemma-4-e2b-deception-behavior-completions) — companion 910-row deception/behavior corpus

### Source repo (full reproducibility)

[`SolshineCode/deception-nanochat-sae-research`](https://github.com/SolshineCode/deception-nanochat-sae-research) — every script, every prompt, every result.

### Release artifacts in this repo

- `README.md` — what you're reading
- `MODEL_CARD_AV.md` — the AV HuggingFace card
- `MODEL_CARD_AR.md` — the AR HuggingFace card
- `lesswrong_post.md` — the announcement post
- `figures/` — 6 release figures (training loss, cos distribution, source breakdown, label word counts, v0.0.1 vs v0.1.0, per-row scatter)
- `examples/round_trip_inference.py` — minimum-viable round-trip inference example

---

AV and AR are published as separate HuggingFace repos. This is the HF convention. Each `repo_id` is a single model. Anthropic's `kitft` team publishes their AV and AR halves the same way (one repo each, cross-linked in READMEs). The pair is logically one artifact for the round-trip eval, but the file layout is two adapters + one linear head + matched sidecars. The READMEs cross-reference each other, and this `nla-gemma-4-e2b` repo is the single landing page that points to both halves.

If you want both in one place: clone the two HF repos with `git lfs install && git clone https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1 && git clone https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`. Both fit on consumer hardware.

---

## Honest performance summary

| Metric | Value |
|---|---|
| Round-trip cosine similarity (mean) | **0.438** ± 0.054 |
| Round-trip cosine similarity (median) | 0.434 |
| Round-trip MSE (mean) | 1.124 (vs random baseline 2.0) |
| Rows above 0.30 noise-floor | **42 / 42 (100%)** |
| n_evaluated | 42 of 50 attempted. 8 produced empty AV outputs and were excluded |
| Min row cos | 0.313 |
| Max row cos | 0.558 |

Cos = 0.438 is well below Anthropic's published 7B numbers (~0.7+). This is the methodology-validation small-model variant. The contribution is the **democratization path**. NLA-style interpretability becomes accessible to researchers without cluster compute.

**Honest failure-rate disclosure.** 16% of attempted eval rows (8 of 50) produced empty AV outputs and were excluded from the cos calculation. That's a real failure mode of the small-model variant at eval time, not a quirk of the held-out set. v0.1.x with the diversified 9-source-family corpus and a longer SFT step budget is the test of whether scale fixes it.

---

## How v0.0.1 fits the constraints (the tricks)

The "trained on a 4 GB laptop" hook is real but it leans on a stack of small descopes, each of which is named below. None of them break the methodology. Together they fit the budget.

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
- **Sidecar `nla_meta.yaml` per checkpoint.** Eval-provenance lookup in one file open instead of grepping result JSONs.

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
- **Honest-accuracy training-trend verdict (slope < -0.002/step AND R² >= 0.10).** Caught a real over-claim before it shipped.
- **Data-permanence directive.** Commit every parquet to `results/` immediately; "regeneratable from script" doesn't count.
- **Eval-provenance sidecar convention.** Commit SHA + parquet SHA-256 + headline cos numbers in YAML alongside each checkpoint.

### Software / tooling (Windows-specific gotchas)

- **`shutil.which("gemini")`** to resolve npm `.CMD` shims. `subprocess.run(["gemini", ...])` can't find them; `subprocess.run([shutil.which("gemini"), ...])` does.
- **`MSYS_NO_PATHCONV=1 taskkill /F /T /PID`** to kill stuck Python processes from Git Bash without MSYS rewriting `/F` to a Windows path.
- **`device_map={"": torch.cuda.current_device()}` (integer)** not `{"": "cuda"}` (string). The string form silently falls back to CPU on bitsandbytes 0.49.2 with no error.
- **`KMP_DUPLICATE_LIB_OK=TRUE`** prefix on every run to dodge the torch+numpy OpenMP conflict on Windows.

**Total spend for v0.0.1:** ~$0.50 in API charges + $0 cloud + electricity for ~6 GPU-hours on a laptop. Time: 3 weeks of evenings, with the failed continuation experiment and the AR-descending false alarm both folded in.

---

## Quick start

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B",
    quantization_config=bnb,
    device_map={"": torch.cuda.current_device()},
)
av = PeftModel.from_pretrained(base, "Solshine/gemma-4-e2b-nla-L23-av-v0_0_1")
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")

# To produce explanations from activations, see examples/round_trip_inference.py
```

For the full inference pipeline (load AV+AR, extract activation at L23, generate explanation, reconstruct activation, measure cos), see `examples/round_trip_inference.py`.

---

## Citation

```bibtex
@misc{gemma4_e2b_nla_v0_0_1,
  title  = {NLA-Gemma-4-E2B (v0.0.1): the first consumer-GPU-trainable open NLA},
  author = {DeLeeuw, Caleb},
  year   = {2026},
  month  = {may},
  url    = {https://github.com/SolshineCode/nla-gemma-4-e2b}
}
```

Please also cite the upstream NLA methodology:

- Fraser-Taliente, K., et al. (2026). *Natural Language Autoencoders*. https://transformer-circuits.pub/2026/nla/
- [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders) — Anthropic's official reference NLA pipeline

---

## Acknowledgments

This release follows Anthropic's NLA methodology exactly, descoped for consumer-GPU compute. Thanks to Kit Fraser-Taliente and the Anthropic interpretability team for publishing the kitft repository as open source. Any errors in the descope are mine.

---

## License

CC BY 4.0 for model weights, datasets, and documentation. Source code in the research repo follows that repo's license.

Caleb DeLeeuw / SolshineCode
