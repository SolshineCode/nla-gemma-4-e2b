"""
Ceiling test: is document-discriminative content even PRESENT in the raw L23
activation, before any AV touches it?

This is the decisive experiment for "how content-discriminative the NLA is." The
AV-output retrieval eval showed the AV TEXT is at chance. That has two very
different explanations:
  (A) the L23 activation encodes doc-discriminative content, but the AV fails to
      surface it  -> the AV is the bottleneck, fixable with better training.
  (B) the L23 activation does not linearly encode doc identity at this scale  ->
      no AV could surface it; the ceiling itself is low.

We measure the ceiling directly: doc-level retrieval on the RAW activations
(same metric as the AV-output eval, so they are directly comparable), plus a
cross-validated linear probe. Run on both the 50-row / 13-doc set (matches the AV
eval) and the full 652-row / 163-doc held-out set (power).

Pure CPU, uses the activations already stored in rl.parquet.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(REPO, "data", "stage1", "rl.parquet")
OUT = os.path.join(REPO, "results", "content_aware_eval", "activation_ceiling.json")
RNG = np.random.RandomState(0)
N_PERM = 5000


def cosine_sim(acts):
    A = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-9)
    sim = A @ A.T
    np.fill_diagonal(sim, -np.inf)  # exclude self
    return sim


def doc_retrieval_sim(sim, doc):
    """Doc-level top-1 from a precomputed similarity matrix and a label vector.
    The sim matrix is label-independent, so permutations only reshuffle `doc`."""
    n = sim.shape[0]
    docs = sorted(set(doc))
    cols = {d: [j for j in range(n) if doc[j] == d] for d in docs}
    didx = {d: k for k, d in enumerate(docs)}
    top1 = 0
    for i in range(n):
        scores = np.array([sim[i, cols[d]].max() if cols[d] else -np.inf for d in docs])
        if int(np.argmax(scores)) == didx[doc[i]] and len(cols[doc[i]]) > 1:
            top1 += 1
    return top1 / n, len(docs)


def perm_test(acts, doc):
    sim = cosine_sim(acts)  # computed ONCE
    obs, nd = doc_retrieval_sim(sim, doc)
    null = np.empty(N_PERM)
    d = np.array(doc, dtype=object)
    for m in range(N_PERM):
        null[m] = doc_retrieval_sim(sim, list(RNG.permutation(d)))[0]
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    z = float((obs - null.mean()) / (null.std() + 1e-12))
    return {"doc_top1": obs, "n_docs": nd, "chance_top1": 1.0 / nd,
            "null_mean": float(null.mean()), "p_value": p, "z": z}


def linear_probe(acts, doc):
    """Cross-validated logistic-regression doc classification. Only docs with >= 3
    rows are usable for stratified CV; report on that subset + the chance line."""
    doc = np.array(doc)
    counts = pd.Series(doc).value_counts()
    keep_docs = counts[counts >= 3].index
    mask = np.isin(doc, keep_docs)
    X, y = acts[mask], doc[mask]
    if len(set(y)) < 2:
        return {"note": "not enough multi-row docs for CV"}
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    k = int(min(5, pd.Series(y).value_counts().min()))
    cv = StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xs, y, cv=cv)
    return {"cv_accuracy_mean": float(scores.mean()), "cv_accuracy_std": float(scores.std()),
            "n_docs_in_probe": int(len(set(y))), "n_samples": int(len(y)),
            "chance": float(1.0 / len(set(y)))}


def run(acts, doc, label):
    ret = perm_test(acts, doc)
    probe = linear_probe(acts, doc)
    print(f"[{label}] activation doc-retrieval top1={ret['doc_top1']:.3f} "
          f"(chance {ret['chance_top1']:.3f}, p={ret['p_value']:.4f}, z={ret['z']:.1f}) | "
          f"linear-probe CV acc={probe.get('cv_accuracy_mean', float('nan')):.3f} "
          f"(chance {probe.get('chance', float('nan')):.3f})")
    return {"retrieval": ret, "linear_probe": probe}


def main():
    df = pd.read_parquet(RL)
    acts_all = np.stack([np.asarray(v, dtype=np.float32) for v in df["activation_vector"]])
    doc_all = list(df["doc_id"])
    result = {
        "config": {"source": "raw L23 activations from rl.parquet",
                   "n_total": len(df), "n_docs_total": len(set(doc_all)),
                   "metric": "doc-level top-1 retrieval (cosine) + logistic-regression CV",
                   "n_perm": N_PERM},
        "set_50_13docs": run(acts_all[:50], doc_all[:50], "n=50 / 13 docs"),
        "set_full": run(acts_all, doc_all, f"n={len(df)} / {len(set(doc_all))} docs"),
    }
    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
