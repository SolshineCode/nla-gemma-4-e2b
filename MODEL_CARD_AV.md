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
- **Batch**: micro_batch=1, grad_accum=16 → effective batch 16
- **Max length**: 512 tokens
- **SFT steps**: 15
- **Hardware**: single 4 GB NVIDIA GTX 1650 Ti Max-Q (laptop)
- **Total wall time**: ~3 GPU-hours end-to-end (including base-model NF4 load)
- **Training corpus**: 2,548 (text, L23 activation, gpt-4o-mini-labeled explanation) triples on the v0.0.x baseline pipeline

## Headline numbers (v0.0.1)

- **Round-trip cosine** (paired with [`Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1)): **0.438 ± 0.054** on n=42 effective held-out activations, 100% above the 0.30 noise floor.
- **AV under SFT loss slope** at converged: −0.0028/step linear regression on raw loss (descending verdict, R² ≥ 0.10).

## What makes this release distinctive

- **First non-Anthropic-team open-source NLA AV** at any model scale. As of 2026-05, every other NLA on HuggingFace Hub is under the `kitft` account (Kit Fraser-Taliente, the paper's first author and Anthropic's official reference). v0.0.1 is the second-source replication.
- **First LoRA-based NLA AV.** Anthropic's published NLA AVs are full fine-tunes at bf16. This release demonstrates that a **LoRA adapter (r=64, α=128)** over NF4-quantized Gemma-4-E2B can train the AV half of an NLA pair to the same output FORMAT class (fluent multi-paragraph descriptive text) at 13× smaller parameter scale. Per-row content fidelity is lower than Anthropic's deployed NLAs — see "Limitations" below for a 10-row Neuronpedia head-to-head. Shipping as a LoRA adapter means the AV loads in ~1.5 GB VRAM on top of the frozen NF4 base, vs ~12 GB for a full bf16 AV.
- **Consumer-GPU trainable.** End-to-end training fits on a 4 GB laptop GPU because of the LoRA + NF4 stack. The methodology descope (NF4 + LoRA + small corpus + ≤300 SFT steps vs Anthropic's full bf16 fine-tune on 8–64 H100s) is documented per parameter.
- **Full open reproducibility chain** in the bundled repo: Stage 0 (extraction) → Stage 1 (split) → Stage 2 (LLM-judge labeling) → Stage 3 (training-format build) → SFT → eval.

## Limitations

**NLAs can produce unexpected or incorrect explanations.** Specifically for this release:

- **Fluent multi-paragraph descriptive output, with lower per-row content fidelity than Anthropic's deployed NLAs.** The AV produces well-formed paragraph-length descriptions in the same FORMAT class as Anthropic's published NLAs. On a 10-row head-to-head against Anthropic's Gemma-3-27B Layer 41 NLA via the Neuronpedia API, Anthropic's NLA more accurately names specific people / events / topics (e.g. "Hillary Clinton's primary momentum", "Obama and Obamacare") where this AV produces more generic linguistic-feature descriptions (e.g. "country-specific statistical weights", "non-binary identity"). The format match is real; per-row content fidelity is meaningfully lower. Comparison data + LLM-judge scores in the source research repo.
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
