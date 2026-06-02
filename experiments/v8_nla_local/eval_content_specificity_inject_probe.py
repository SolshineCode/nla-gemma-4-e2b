"""
Single-token forced-choice content probe (GPU, constrained-decoding).

Bypasses the AV's free-form template entirely. Inject each activation into the AV
prompt, then score a fixed multiple-choice continuation ("This vector is about
<topic>") and ask: does injecting THIS activation make the model prefer THIS
document's topic over distractor topics?

Two readouts per row (4-way, chance = 0.25):
  - simple: argmax over candidates of length-normalized logprob with the activation
    injected.
  - shift (primary): argmax over candidates of (logprob | inject a_i) minus
    (logprob | inject mean-activation baseline), length-normalized. The baseline
    subtraction removes topic-frequency confounds and isolates the activation's
    causal effect on topic preference. This is what makes a null trustworthy.

Degeneracy guard: report the distribution of chosen positions. A null is only
meaningful if choices are spread (not "always option 1", which would be a
format artifact rather than a content result).

Reuses the exact injection mechanism from examples/round_trip_example.py
(inject at the U+3297 slot at scale sqrt(d_model)).
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from scipy import stats

REPO = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(REPO, "data", "stage1", "rl.parquet")
AV_LOCAL = os.path.join(REPO, "checkpoints", "av_v0_1_dd_inj39_bf16_long", "step_000250")
OUT = os.path.join(REPO, "results", "content_aware_eval",
                   "content_specificity_inject_probe.json")
BASE = "google/gemma-4-E2B"
LAYER = 23
D_MODEL = 1536
INJ_ID, LEFT_ID, RIGHT_ID = 249568, 236813, 954
INJ_CHAR = chr(0x3297)
AV_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "activation vectors from a language model. Your overall task is to describe the "
    "semantic content of that activation vector.\n\nWe will pass the vector enclosed "
    "in <concept> tags into your context. You must then produce an explanation for "
    "the vector, enclosed within <explanation> tags. The explanation consists of 2-3 "
    "text snippets describing that vector.\n\nHere is the vector:\n\n<concept>"
    + INJ_CHAR + "</concept>"
)
# Deterministic, objective topic labels for the 13 held-out documents.
TOPICS = {
    "doc_00000001": "Hillary Clinton's campaign",
    "doc_00000002": "a political opinion column",
    "doc_00000017": "NFL football",
    "doc_00000018": "a casino blackjack game",
    "doc_00000019": "luxury cars in rap music",
    "doc_00000022": "space exploration",
    "doc_00000028": "President Obama",
    "doc_00000035": "airstrikes on ISIS in Syria",
    "doc_00000047": "the Uber app in Dubai",
    "doc_00000048": "Israel and China",
    "doc_00000054": "a state Supreme Court appointment",
    "doc_00000062": "the Snowden surveillance leaks",
    "doc_00000065": "the Evil Dead horror game",
}
K = 4
RNG = np.random.RandomState(0)


def bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)


def scale(v):
    v = np.asarray(v, dtype=np.float32)
    return (v / (np.linalg.norm(v) + 1e-9) * np.sqrt(D_MODEL)).astype(np.float32)


def main():
    df = pd.read_parquet(RL)
    n = 50
    acts = [np.asarray(df["activation_vector"].iloc[i], dtype=np.float32) for i in range(n)]
    scaled = [scale(a) for a in acts]
    baseline = scale(np.mean(np.stack(acts), axis=0))  # content-neutral reference
    doc = [df["doc_id"].iloc[i] for i in range(n)]

    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb(), device_map={"": torch.cuda.current_device()})
    src = AV_LOCAL if os.path.isdir(AV_LOCAL) else "Solshine/gemma-4-e2b-nla-L23-av-v0_1_dd-step_250"
    av = PeftModel.from_pretrained(base, src)
    av.eval()
    print("loaded AV from", src)

    pending = {"input_ids": None, "vec": None}

    def hook(module, inp, out):
        if out.shape[1] <= 1:
            return out
        ids, vec = pending["input_ids"], pending["vec"]
        if ids is None or vec is None:
            return out
        h = out.clone()
        for p2 in range(1, ids.shape[1] - 1):
            if (ids[0, p2].item() == INJ_ID and ids[0, p2 - 1].item() == LEFT_ID
                    and ids[0, p2 + 1].item() == RIGHT_ID):
                h[0, p2] = vec[0].to(h.dtype)
                break
        return h

    handle = av.get_input_embeddings().register_forward_hook(hook)

    prefix_ids = tok.encode(AV_TEMPLATE + "\n\n<explanation> This vector is about",
                            return_tensors="pt")
    plen = prefix_ids.shape[1]

    def score(cont_ids, vec):
        full = torch.cat([prefix_ids, cont_ids], dim=1).to(av.device)
        pending["input_ids"] = full
        pending["vec"] = torch.from_numpy(vec).to(av.device).unsqueeze(0)
        with torch.no_grad():
            logits = av(input_ids=full).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        tot = 0.0
        m = cont_ids.shape[1]
        for k in range(m):
            tot += lp[plen + k - 1, full[0, plen + k].item()].item()
        return tot / max(m, 1)  # length-normalized (per-token mean logprob)

    all_topics = sorted(set(TOPICS.values()))
    rows = []
    try:
        for i in range(n):
            true_t = TOPICS[doc[i]]
            distract = [t for t in all_topics if t != true_t]
            chosen = list(RNG.choice(distract, K - 1, replace=False)) + [true_t]
            RNG.shuffle(chosen)
            true_pos = chosen.index(true_t)
            conts = [tok.encode(" " + t, return_tensors="pt", add_special_tokens=False)
                     for t in chosen]
            lp_inj = [score(c, scaled[i]) for c in conts]
            lp_base = [score(c, baseline) for c in conts]
            shift = [lp_inj[k] - lp_base[k] for k in range(K)]
            rows.append({
                "row": i, "doc": doc[i], "true_pos": true_pos,
                "pred_simple": int(np.argmax(lp_inj)),
                "pred_shift": int(np.argmax(shift)),
                "lp_inj": lp_inj, "shift": shift, "candidates": chosen,
            })
            if i % 10 == 0:
                print(f"row {i} done")
    finally:
        handle.remove()

    simple_acc = np.mean([r["pred_simple"] == r["true_pos"] for r in rows])
    shift_acc = np.mean([r["pred_shift"] == r["true_pos"] for r in rows])
    simple_c = sum(r["pred_simple"] == r["true_pos"] for r in rows)
    shift_c = sum(r["pred_shift"] == r["true_pos"] for r in rows)
    p_simple = stats.binomtest(simple_c, n, 1.0 / K, alternative="greater").pvalue
    p_shift = stats.binomtest(shift_c, n, 1.0 / K, alternative="greater").pvalue
    from collections import Counter
    result = {
        "config": {"av": src, "n": n, "k_choices": K, "chance": 1.0 / K,
                   "metric": "length-normalized continuation logprob; shift = inject - mean-baseline"},
        "simple_inject": {"correct": simple_c, "acc": float(simple_acc), "binom_p": p_simple},
        "shift_vs_baseline": {"correct": shift_c, "acc": float(shift_acc), "binom_p": p_shift},
        "degeneracy_check": {
            "pred_simple_position_dist": dict(Counter(r["pred_simple"] for r in rows)),
            "pred_shift_position_dist": dict(Counter(r["pred_shift"] for r in rows)),
        },
        "verdict_shift": "content signal (p<0.05)" if p_shift < 0.05 else "no signal above chance",
        "per_row": rows,
    }
    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nsimple inject acc {simple_c}/{n}={simple_acc:.3f} (chance {1/K:.2f}) p={p_simple:.3f}")
    print(f"shift-vs-baseline acc {shift_c}/{n}={shift_acc:.3f} p={p_shift:.3f} -> {result['verdict_shift']}")
    print("pred_shift positions:", result["degeneracy_check"]["pred_shift_position_dist"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
