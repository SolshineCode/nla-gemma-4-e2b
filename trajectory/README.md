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
  - gemma
  - consumer-gpu
  - peft
  - lora
  - training-trajectory
  - research-artifact
library_name: peft
pipeline_tag: text-generation
---

# Gemma-4-E2B NLA AV (Actor) — v0.1.x cheap-path training trajectory

> Open research-data release: every 50-step checkpoint from a 500-step SFT run on a 4 GB GTX 1650 Ti Max-Q. Includes the full training trajectory, NOT just the final adapter, so researchers can study how the model learned (or did not learn) across the regime.

This is a **trajectory release** — a set of intermediate AV (Actor) LoRA adapters at every 50 steps of training. Each checkpoint lives in its own subdirectory.

For the single "use this one" final adapter, see [`Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-cheap-path`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-cheap-path) (added after training completes).

For the matched AR (Critic) — currently the same v0.0.1 AR is paired with these AVs. A matched v0.1.x AR is in the queue for a future release.

## Why a trajectory release

After the v0.0.1 NLA pair shipped, a 5-experiment investigation surfaced that the round-trip cosine similarity 0.438 reported on v0.0.1 was principally a structural AR projection (~95% content-independent), not a measurement of explanation faithfulness. Two follow-up ablations (H13 and H14) established that within the v0.0.x compute regime (4 GB GPU + ~700-2000-row corpus + ≤60 SFT steps), neither label quality nor step count alone moves the AV_OUT−EMPTY mean delta off the +0.018-0.024 plateau.

The v0.1.x cheap-path is the **scale-up experiment within the 4 GB regime**: a 500-step run on the 4,734-row Gemini persona+audit corpus (6.8× more data, 8× more SFT steps than v0.0.x). The trajectory release lets the community:

1. Run their own H5-style ablation on any checkpoint
2. See the per-checkpoint loss + content-sensitivity trend
3. Identify which checkpoint (if any) breaks the v0.0.x ceiling
4. Pull whichever intermediate adapter best suits their downstream use case
5. Verify or reproduce the trajectory-aware findings

Even if the final adapter is no more faithful than v0.0.1, the trajectory itself is a citable research artifact about how small-model NLAs train under aggressive descope.

## Training setup

- **Base model**: `google/gemma-4-E2B` (2B params, 35 text layers)
- **Layer**: L23 (~2/3 through the text-layer stack)
- **Quantization**: NF4 4-bit base weights + bf16 LoRA adapters
- **LoRA**: r=64, alpha=128, target modules same regex as v0.0.1 (language-model layers only, excludes audio tower)
- **Injection convention**: Forward-hook on embedding layer
- **Training corpus**: 4,734 rows from `experiments/v8_nla_local/data/stage3_v0_1_full/av_sft.parquet` — 9 source families (Wikipedia, FineWeb-Edu, arXiv, in-repo Gemma-4-E2B deception completions, PKU-SafeRLHF, Anthropic/discrim-eval, Anthropic/persuasion, CAI harmless, Anthropic/llm_global_opinions), labeled by Gemini CLI with the Dr Chen + Dr Otsuka persona+audit pipeline
- **Optimizer**: AdamW 8-bit, lr=1e-4 (no decay schedule)
- **Batch**: micro_batch=1, grad_accum=16 → effective batch 16
- **Max length**: 512 tokens
- **Hardware**: 4 GB GTX 1650 Ti Max-Q laptop
- **Total wall time**: ~25 GPU-hours
- **Training script**: `experiments/v8_nla_local/stage_av_sft.py` in the source repo

## Layout (filled in as training progresses)

```
step_000050/   adapter at  50 SFT steps (50/500 = 10%)
step_000100/   adapter at 100 SFT steps (20%)
step_000150/   adapter at 150 SFT steps (30%)
step_000200/   adapter at 200 SFT steps (40%) — eval ablation snapshot
step_000250/   adapter at 250 SFT steps (50%)
step_000300/   adapter at 300 SFT steps (60%)
step_000350/   adapter at 350 SFT steps (70%)
step_000400/   adapter at 400 SFT steps (80%)
step_000450/   adapter at 450 SFT steps (90%)
step_000500/   adapter at 500 SFT steps (100% — final)
```

Each subdirectory contains `adapter_config.json` + `adapter_model.safetensors` + a per-checkpoint `nla_meta.yaml` sidecar with the loss at that step, training parameters, and (where available) the eval-provenance block from an H5-style ablation against that checkpoint.

## Loading any single checkpoint

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B", quantization_config=bnb,
                                              device_map={"": torch.cuda.current_device()})
# Pull a specific checkpoint via subfolder:
av = PeftModel.from_pretrained(
    base,
    "Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory",
    subfolder="step_000100",
)
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
```

## Loss trajectory (live, updated as training progresses)

| Step | Loss | Notes |
|---|---|---|
| 1 | 3.74 | initial |
| 50 | 2.38 | first checkpoint |
| 100 | 2.24 | published in this release |
| 150 | (pending) | |
| 200 | (pending) | first scheduled eval ablation snapshot |
| 250 | (pending) | |
| 300 | (pending) | |
| 350 | (pending) | |
| 400 | (pending) | |
| 450 | (pending) | |
| 500 | (pending) | final |

## H5-style content-sensitivity per checkpoint (live, updated as eval ablations land)

| Step | AV_OUT cos | EMPTY cos | Δ (AV_OUT − EMPTY) | Above 0.30 floor |
|---|---|---|---|---|
| 200 | (pending) | (pending) | (pending) | (pending) |
| 500 | (pending) | (pending) | (pending) | (pending) |

(Other intermediate ablations will be added if the AV_OUT−EMPTY delta moves substantially.)

## Citation

```bibtex
@misc{gemma4_e2b_nla_v0_1_x_trajectory,
  title  = {Gemma-4-E2B NLA AV training trajectory (v0.1.x cheap-path): 10 checkpoints across 500 SFT steps on a 4 GB consumer GPU},
  author = {DeLeeuw, Caleb},
  year   = {2026},
  month  = {may},
  url    = {https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory}
}
```

Please also cite the upstream NLA methodology:

- Fraser-Taliente, K., et al. (2026). *Natural Language Autoencoders*. https://transformer-circuits.pub/2026/nla/

## See also

- v0.0.1 final AV: [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1)
- v0.0.1 matched AR: [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1)
- v0.1.x cheap-path final (after training): [`Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-cheap-path`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-cheap-path)
- Public bundled release: [`SolshineCode/nla-gemma-4-e2b`](https://github.com/SolshineCode/nla-gemma-4-e2b)
- Source research repo: `SolshineCode/deception-nanochat-sae-research` — currently private, **available upon request — DM me**
