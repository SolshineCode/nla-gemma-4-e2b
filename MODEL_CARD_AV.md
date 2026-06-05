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
library_name: peft
pipeline_tag: text-generation
---

# Gemma-4-E2B NLA AV (Activation Verbalizer) — v0.0.1

LoRA adapter for `google/gemma-4-E2B` that takes a 1536-dimensional residual-stream activation captured at layer 23 and produces a natural-language explanation of what the activation represents.

This is the **first non-Anthropic-team open-source NLA Activation Verbalizer** released publicly. Trained end-to-end on a single 4 GB consumer GPU (NVIDIA GTX 1650 Ti Max-Q) following the methodology of Fraser-Taliente, Kantamneni, Ong et al. 2026 ([Transformer Circuits](https://transformer-circuits.pub/2026/nla/)).

Pairs with the matched [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1) reconstructor.

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
import torch

BASE = "google/gemma-4-E2B"
AV_REPO = "Solshine/gemma-4-e2b-nla-L23-av-v0_0_1"

# Injection convention
INJECTION_TOKEN_ID = 249568           # ㊗
INJECTION_LEFT_NEIGHBOR_ID = 236813   # <
INJECTION_RIGHT_NEIGHBOR_ID = 954     # >
INJECTION_CHAR = chr(0x3297)
D_MODEL = 1536
INJECTION_SCALE = float(np.sqrt(D_MODEL))  # = 39.2; matches Gemma-4-E2B token-embed norm

PROMPT = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. "
    "You must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets "
    "describing that vector.\n\nHere is the vector:\n\n"
    f"<concept>{INJECTION_CHAR}</concept>"
)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
av = PeftModel.from_pretrained(base, AV_REPO); av.eval()

# At inference: hook the embedding layer to replace ㊗'s embedding with the
# scaled activation vector when the [<, ㊗, >] trio is detected.
pending = {"input_ids": None, "vec": None}
def hook(module, args_in, output):
    if output.shape[1] <= 1: return output
    ids = pending["input_ids"]; vec = pending["vec"]
    if ids is None or vec is None: return output
    h = output.clone()
    for b in range(ids.shape[0]):
        for p in range(1, ids.shape[1] - 1):
            if (ids[b,p].item() == INJECTION_TOKEN_ID
                and ids[b,p-1].item() == INJECTION_LEFT_NEIGHBOR_ID
                and ids[b,p+1].item() == INJECTION_RIGHT_NEIGHBOR_ID):
                h[b,p] = vec[b].to(h.dtype); break
    return h
av.get_input_embeddings().register_forward_hook(hook)

# Use
activation_vector = np.random.randn(D_MODEL).astype(np.float32)  # your 1536-d L23 activation
scaled = activation_vector / (np.linalg.norm(activation_vector) + 1e-9) * INJECTION_SCALE
ids = tok.encode(PROMPT, return_tensors="pt").to(av.device)
pending["input_ids"] = ids
pending["vec"] = torch.from_numpy(scaled).to(av.device).unsqueeze(0)

with torch.no_grad():
    out = av.generate(input_ids=ids, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

Working end-to-end round-trip example with the matched AR: `examples/round_trip_example.py` in the [bundled public repo](https://github.com/SolshineCode/nla-gemma-4-e2b).

## Training setup

- **Base model**: `google/gemma-4-E2B` (2B parameters, 35 text layers)
- **Activation layer**: L23 residual stream
- **Quantization**: NF4 4-bit base weights + fp16 LoRA adapters
- **LoRA config**: r=64, α=128, target modules = `model.language_model.layers.\d+.(self_attn|mlp).(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)` (language-model layers only; excludes audio tower)
- **Injection mechanism**: forward hook on embedding layer; replaces ㊗ token's embedding with the L2-normalized activation rescaled to `injection_scale = sqrt(d_model) = 39.2` (matches the empirically-measured Gemma-4-E2B token-embedding norm of 39.25)
- **Optimizer**: AdamW 8-bit, lr=1e-4
- **Batch**: micro_batch=1, grad_accum=4 → effective batch 4
- **Max length**: 512 tokens
- **SFT steps**: 55 total (15 base + 40 continuation)
- **Hardware**: single 4 GB NVIDIA GTX 1650 Ti Max-Q (laptop)
- **Total wall time**: ~3 GPU-hours end-to-end (including base-model NF4 load)
- **Training corpus**: 2,548 (text, L23 activation, gpt-4o-mini-labeled explanation) triples on the v0.0.x baseline pipeline

## Headline numbers (v0.0.1)

- **Round-trip cosine** (paired with [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1)): **0.438 ± 0.054** on n=42 effective held-out activations, 100% above the 0.30 noise floor.
- **AV under SFT loss slope** at converged: −0.0028/step linear regression on raw loss (descending verdict, R² ≥ 0.10).

## Evaluation across released versions

![NLA AV evaluation across released versions](figures/nla_eval_across_versions.png)

Content-fidelity doc-level retrieval (left) and reconstruction round-trip cosine (right) across the
released AV versions. The verbalizer's content-surfacing is domain-sensitive: at chance on
out-of-domain news, modestly but significantly above chance in-domain (see Limitations). Round-trip
cosine is shown with the caveat that it is structural-projection dominated, not a faithfulness
metric. Regenerate with `make_nla_eval_figure.py` as new versions or evaluations land.

## What makes this release distinctive

- **First non-Anthropic-team open-source NLA AV** at any model scale. As of 2026-05, every other NLA on HuggingFace Hub is under the `kitft` account (Kit Fraser-Taliente, the paper's first author and Anthropic's official reference). v0.0.1 is the second-source replication.
- **First LoRA-based NLA AV.** Anthropic's published NLA AVs are full fine-tunes at bf16. This release demonstrates that a **LoRA adapter (r=64, α=128)** over NF4-quantized Gemma-4-E2B can train the AV half of an NLA pair to the same output FORMAT class (fluent multi-paragraph descriptive text) at 13× smaller parameter scale. Per-row content fidelity is lower than Anthropic's deployed NLAs, though a ceiling test shows that content is present in the activation (60% linear probe on 13-way document identity) and simply not yet surfaced by the verbalizer rather than absent — see "Limitations" below for the head-to-head and the activation-ceiling result. Shipping as a LoRA adapter means the AV loads in ~1.5 GB VRAM on top of the frozen NF4 base, vs ~12 GB for a full bf16 AV.
- **Consumer-GPU trainable.** End-to-end training fits on a 4 GB laptop GPU because of the LoRA + NF4 stack. The methodology descope (NF4 + LoRA + small corpus + ≤300 SFT steps vs Anthropic's full bf16 fine-tune on 8–64 H100s) is documented per parameter.
- **Full open reproducibility chain** in the bundled repo: Stage 0 (extraction) → Stage 1 (split) → Stage 2 (LLM-judge labeling) → Stage 3 (training-format build) → SFT → eval.

## Release rationale: why this SFT pair and not a GRPO checkpoint

The Anthropic NLA recipe (Fraser-Taliente et al. 2026) has four phases: Stages 0–3 (data + labeling) → SFT (supervised fine-tune of the AV+AR pair) → **Phase 4 GRPO** (joint REINFORCE-style RL fine-tune of the AV with the AR's reconstruction-MSE as reward signal, plus an AR "keep-up" SFT update and a KL anchor). The published `v0.0.1` and `v0.1` pairs are the **SFT-only** output of Phases 1–3; Phase 4 GRPO was deferred at first release because it had not yet been adapted to the 4 GB hardware regime.

Between 2026-05-25 and 2026-05-29 the deferred Phase 4 was implemented and run **end-to-end on the same 4 GB GTX 1650 Ti Max-Q**, with alternating AV/AR loads and R=4 rollout batching to fit in VRAM. The trial swept **5 reward formulations × 4 entropy regimes across 120 rollouts**, with intermediate L2 cross-row-argmax readouts at rollouts 40, 60, 80, 100, 120:

| Rollout | Reward | Entropy β | L2 cross-row argmax (n=10) | AV output quality |
|---:|---|---:|---:|---|
| 40 | MSE | 0.0 | 0.100 (chance) | coherent multi-paragraph (same class as SFT v0.1) |
| 60 | MSE | 0.3 | 0.100 (chance) | random Unicode tokens — degenerate |
| 80 | contrastive-mean | 1.0 | 0.100 (chance) | whitespace-only — degenerate |
| 100 | contrastive-max | 1.0 | 0.100 (chance) | "evasion evasion evasion …" mode collapse |
| 120 | contrastive-max + AR-contrastive | 0.1 | 0.100 (chance) | "evasion evasion evasion …" mode collapse |

**Verdict.** No GRPO checkpoint is shipped:
- **r40** (the only checkpoint with intact AV-output coherence) matched the SFT v0.1 L2 margin within noise — it did not beat the released SFT pair on the headline metric, so shipping it would add nothing.
- **r60–r120** (all higher-entropy configurations) produced AV output that is unusable for any downstream consumer of the NLA — random tokens, whitespace, or the "evasion" attractor. These checkpoints are research-valuable but unfit to ship as an interpretability tool.

**The released SFT pair is strictly better than any GRPO checkpoint we produced on this hardware**: both classes are at L2 = chance on per-row identity, but the released SFT pair preserves the coherent multi-paragraph descriptive output that gives the NLA pipeline its interpretability surface, whereas the high-entropy GRPO checkpoints destroyed that surface without compensating with any measurable per-row-fidelity gain.

**Research contribution.** This trial closed the scope of the SFT-only "ceiling" framing: combining the 8-attempt SFT lever sweep with the 5-readout GRPO sweep yields **14 distinct training attempts spanning the full Anthropic recipe**, all converging to L2 = chance at 4 GB. The L2 ceiling at this hardware scale is therefore robust to (a) optimizer-/loss-/scheduler-side levers within SFT, (b) reward shape (MSE vs contrastive vs contrastive-max), (c) entropy regularization (β ∈ {0, 0.1, 0.3, 1.0}), and (d) training paradigm (SFT-only vs SFT+GRPO). The open question — whether the bottleneck is **base-model scale** (2B vs 27B/70B) or **the 4 GB hardware constraint** (NF4 + LoRA + small contrast pool) — would be answered by a cross-model recipe-controlled retrain on Gemma-3-27B; that experiment is flagged for follow-on grant-funded work.

The v0.0.1 + v0.1 SFT pair on this repo therefore represents the **best-coherent-output checkpoint** from a comprehensive characterization of the Anthropic NLA recipe at 4 GB, **not** a checkpoint that ran out of training budget before further phases could be attempted.

## Limitations

**NLAs can produce unexpected or incorrect explanations.** Specifically for this release:

- **Fluent multi-paragraph descriptive output, with lower per-row content fidelity than Anthropic's deployed NLAs.** The AV produces well-formed paragraph-length descriptions in the same FORMAT class as Anthropic's published NLAs. On a 10-row head-to-head against Anthropic's Gemma-3-27B Layer 41 NLA via the Neuronpedia API, Anthropic's NLA more accurately names specific people / events / topics (e.g. "Hillary Clinton's primary momentum", "Obama and Obamacare") where this AV produces more generic linguistic-feature descriptions (e.g. "country-specific statistical weights", "non-binary identity"). The format match is real; per-row content fidelity is meaningfully lower. A direct content-specificity retrieval eval (does each AV output recover its own source document?) puts this AV **at chance on out-of-domain text** (politically-themed news, a genre absent from training) across lexical, semantic, and two LLM-judge probes, so on that distribution the output is diverse (45/50 unique strings) but not per-row content- or theme-discriminative. The picture is **domain-sensitive**, though: on held-out **in-domain** text (web text of the kind represented in training), the same AV recovers its own source document modestly but significantly **above chance** — doc-level retrieval ≈ 2× chance (n=50 / 13 docs: 0.14 vs 0.077, p≈0.08; confirmed at n=160 / 40 docs: 0.056 vs 0.025 lexical, p=0.01, and 0.050 vs 0.025 semantic, p=0.03), with some genuinely content-bearing outputs (e.g. naming "1919 Paris Peace Conference" or "filmmaker Nanfu Wang"). One important nuance keeps this honest: a blind reasoning-LLM judge still cannot identify the true source above chance (n=38), so this in-domain advantage reflects **occasional exact content-word surfacing, not systematic conditioning** on the activation — the AV surfaces source content sometimes in-domain and falls back to a learned prior out-of-domain. Importantly, the gap is the verbalizer's, not the activation's: a ceiling test on the raw L23 activations recovers the source document well above chance (doc-level retrieval 0.24 vs 0.077; a logistic probe reads 13-way document identity at 60%), so the content the AV does not yet surface is demonstrably present in the activation. That places the bottleneck in the verbalizer's reading of the injected activation rather than in the 2B model's content — an open problem under active investigation, not a settled matter of training budget — and a layer sweep finds L17 carrying more than 2× the document signal of L23, a lever a future AV can retarget. Comparison data, retrieval-eval scripts, and per-trial LLM-judge data are in the bundled repo under [`experiments/v8_nla_local/`](https://github.com/SolshineCode/nla-gemma-4-e2b/tree/main/experiments/v8_nla_local) (`CONTENT_SPECIFICITY_EVAL.md`).
- **Round-trip cosine has a structural-projection component.** Replicating the published §"Measuring steganography" and §"Characterizing confabulations" tests on v0.0.1: paraphrasing the AV output moves the round-trip cosine by ~3% (Δcos = +0.014); removing entire claims from the AV output moves cosine by ~0% per claim (Δcos = +0.001 per claim). Most of the v0.0.1 round-trip-cosine signal is the AR's structural projection toward "somewhere in OpenWebText L23 activation space," not the explanation's specific content. **Use AV-side per-row content-fidelity judging (validity × specificity × relatedness rubric) alongside round-trip cosine, never round-trip cosine alone.**
- **Template-heavy outputs.** Inspection shows ~80% of held-out-row outputs share a small set of structural templates with content-conditional fill-in slots. Use multiple feature angles + content-judge scoring rather than treating any single output as a verbatim summary of the activation.
- **Hardware-bound quality ceiling.** Numbers reflect a single 4 GB GTX 1650 Ti Max-Q. Larger consumer GPUs with bf16 + full fine-tune + larger corpus would close some of the qualitative gap with the published reference NLAs.

Full development history and methodology retraction notes: [`HISTORY.md`](https://github.com/SolshineCode/nla-gemma-4-e2b/blob/main/HISTORY.md). Internal experiment numbering and audit trail: source research repo (available on request).

## Sidecar (training provenance YAML)

The companion `nla_meta.yaml` records training-time hyperparameters for round-tripping at inference. Read `injection_scale` from this file rather than hardcoding to avoid train-test mismatches.

## Citation

```bibtex
@article{frasertaliente2026nla,
  title={Natural Language Autoencoders},
  author={Fraser-Taliente, Kit and Kantamneni, Kshitij and Ong, Antonia and others},
  journal={Transformer Circuits},
  year={2026},
  url={https://transformer-circuits.pub/2026/nla/}
}

@misc{deleeuw2026nlagemma4e2bav,
  title={Gemma-4-E2B NLA AV (v0.0.1): a 4 GB consumer-GPU Activation Verbalizer},
  author={DeLeeuw, Caleb (SolshineCode)},
  year={2026},
  url={https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1}
}
```

## License

CC-BY 4.0. See [`LICENSE`](LICENSE) in the bundled repo.
