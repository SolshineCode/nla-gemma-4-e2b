"""
Layer sweep: which layer's activation is most document-discriminative?

The NLA reads L23. If L23 itself carries little doc-discriminative content, no AV
reading L23 could surface it, and a different layer might be a better NLA target.
This sweep re-extracts activations at several layers for the same held-out texts
and runs the same doc-level retrieval ceiling at each layer.

For each text we capture the LAST-token residual-stream activation at each swept
layer (same convention as the NLA's extraction), then measure doc-level top-1
retrieval (chance = 1/n_docs) with a permutation null.

GPU. One forward pass per text captures all layers at once via hooks.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(REPO, "data", "stage1", "rl.parquet")
OUT = os.path.join(REPO, "results", "content_aware_eval", "layer_ceiling_sweep.json")
BASE = "google/gemma-4-E2B"
LAYERS = [5, 9, 13, 17, 21, 23, 27, 31]
N_TEXTS = 200          # subset of the 652 for speed; covers many docs
N_PERM = 2000
RNG = np.random.RandomState(0)


def bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)


def cosine_sim(acts):
    A = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-9)
    sim = A @ A.T
    np.fill_diagonal(sim, -np.inf)
    return sim


def doc_top1(sim, doc):
    n = sim.shape[0]
    docs = sorted(set(doc))
    cols = {d: [j for j in range(n) if doc[j] == d] for d in docs}
    didx = {d: k for k, d in enumerate(docs)}
    hit = 0
    for i in range(n):
        scores = np.array([sim[i, cols[d]].max() if cols[d] else -np.inf for d in docs])
        if int(np.argmax(scores)) == didx[doc[i]] and len(cols[doc[i]]) > 1:
            hit += 1
    return hit / n, len(docs)


def perm(sim, doc):
    obs, nd = doc_top1(sim, doc)
    d = np.array(doc, dtype=object)
    null = np.array([doc_top1(sim, list(RNG.permutation(d)))[0] for _ in range(N_PERM)])
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return {"doc_top1": obs, "n_docs": nd, "chance": 1.0 / nd, "p_value": p,
            "z": float((obs - null.mean()) / (null.std() + 1e-12))}


def main():
    df = pd.read_parquet(RL).iloc[:N_TEXTS]
    texts = list(df["detokenized_text_truncated"])
    doc = list(df["doc_id"])
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb(), device_map={"": torch.cuda.current_device()})
    model.eval()
    layers = model.model.language_model.layers

    captured = {}
    handles = []
    for li in LAYERS:
        def mk(li):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[li] = h[:, -1, :].detach().cpu().float().numpy()[0]
            return hook
        handles.append(layers[li].register_forward_hook(mk(li)))

    acts = {li: [] for li in LAYERS}
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            captured.clear()
            model(input_ids=ids)
            for li in LAYERS:
                acts[li].append(captured[li].copy())
    for h in handles:
        h.remove()

    result = {"config": {"layers": LAYERS, "n_texts": len(texts),
                         "n_docs": len(set(doc)), "n_perm": N_PERM,
                         "metric": "last-token doc-level retrieval ceiling per layer"},
              "per_layer": {}}
    for li in LAYERS:
        A = np.stack(acts[li])
        r = perm(cosine_sim(A), doc)
        result["per_layer"][li] = r
        print(f"L{li}: doc_top1={r['doc_top1']:.3f} (chance {r['chance']:.3f}, "
              f"p={r['p_value']:.4f}, z={r['z']:.1f})")
    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
