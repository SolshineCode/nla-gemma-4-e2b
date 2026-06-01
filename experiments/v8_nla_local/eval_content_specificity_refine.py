"""
Refinements to the content-specificity retrieval eval, addressing two ways the
base eval could UNDER-detect signal (giving the 'there is content signal'
hypothesis its strongest shot):

  R1. Local-window source: the activation at the last token encodes mostly LOCAL
      context. Matching against the full truncated doc may dilute it. Re-match
      against the last TAIL_CHARS characters of the source.
  R2. Template stratification: 18/50 outputs collapse to the 'country-specific...'
      prefix. Maybe the OTHER ~32 carry signal. Re-run retrieval on the non-collapsed
      subset only.

Same permutation null. Pure CPU. Semantic + TF-IDF.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(REPO, "data", "stage1", "rl.parquet")
AV = os.path.join(REPO, "results", "content_aware_eval",
                  "av_outputs_v0_1_dd_n50_from_neuronpedia.json")
OUT = os.path.join(REPO, "results", "content_aware_eval",
                   "content_specificity_refine.json")
RNG = np.random.RandomState(0)
N_PERM = 5000
TAIL_CHARS = 300
TEMPLATE_PREFIX = "The model tracks a list of country-"


def doc_top1_and_gap(sim, doc):
    n = sim.shape[0]
    docs = sorted(set(doc))
    cols_by_doc = {d: [j for j in range(n) if doc[j] == d] for d in docs}
    d_index = {d: k for k, d in enumerate(docs)}
    top1 = 0
    for i in range(n):
        scores = np.array([sim[i, cols_by_doc[d]].max() for d in docs])
        if int(np.argmax(scores)) == d_index[doc[i]]:
            top1 += 1
    same, diff = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (same if doc[i] == doc[j] else diff).append(sim[i, j])
    gap = float(np.mean(same) - np.mean(diff))
    return top1 / n, gap, len(docs)


def perm(sim, doc):
    obs_t, obs_g, nd = doc_top1_and_gap(sim, doc)
    nt = np.empty(N_PERM)
    ng = np.empty(N_PERM)
    idx = np.arange(sim.shape[0])
    for m in range(N_PERM):
        p = RNG.permutation(idx)
        nt[m], ng[m], _ = doc_top1_and_gap(sim[p], doc)
    pt = float((np.sum(nt >= obs_t) + 1) / (N_PERM + 1))
    pg = float((np.sum(ng >= obs_g) + 1) / (N_PERM + 1))
    return {"doc_top1": obs_t, "chance_top1": 1.0 / nd, "top1_p": pt,
            "gap": obs_g, "gap_p": pg, "gap_z": float((obs_g - ng.mean()) / (ng.std() + 1e-12))}


def make_sim(av, src, kind):
    if kind == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                              stop_words="english").fit(src + av)
        return cosine_similarity(vec.transform(av), vec.transform(src))
    if kind == "semantic":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ea = model.encode(av, normalize_embeddings=True, show_progress_bar=False)
        es = model.encode(src, normalize_embeddings=True, show_progress_bar=False)
        return ea @ es.T


def main():
    df = pd.read_parquet(RL)
    av_all = json.load(open(AV, encoding="utf-8"))["our_outputs"]
    n = len(av_all)
    full_src = [df["detokenized_text_truncated"].iloc[i] for i in range(n)]
    tail_src = [s[-TAIL_CHARS:] for s in full_src]
    doc = [df["doc_id"].iloc[i] for i in range(n)]
    keep = [i for i in range(n) if not av_all[i].startswith(TEMPLATE_PREFIX)]

    out = {"config": {"n_rows": n, "tail_chars": TAIL_CHARS, "n_perm": N_PERM,
                      "n_non_template": len(keep)}, "results": {}}

    for kind in ["tfidf", "semantic"]:
        # R1: full vs tail-window source
        sim_full = make_sim(av_all, full_src, kind)
        sim_tail = make_sim(av_all, tail_src, kind)
        # R2: non-template subset, tail source
        av_k = [av_all[i] for i in keep]
        tail_k = [tail_src[i] for i in keep]
        doc_k = [doc[i] for i in keep]
        sim_k = make_sim(av_k, tail_k, kind)
        out["results"][kind] = {
            "R1_full_doc": perm(sim_full, doc),
            "R1_tail_window": perm(sim_tail, doc),
            "R2_nontemplate_tail": perm(sim_k, doc_k),
        }
        r = out["results"][kind]
        print(f"[{kind}] full top1={r['R1_full_doc']['doc_top1']:.3f}(p{r['R1_full_doc']['top1_p']:.2f}) "
              f"tail top1={r['R1_tail_window']['doc_top1']:.3f}(p{r['R1_tail_window']['top1_p']:.2f}) "
              f"nontmpl top1={r['R2_nontemplate_tail']['doc_top1']:.3f}(p{r['R2_nontemplate_tail']['top1_p']:.2f}) "
              f"| gap_p full={r['R1_full_doc']['gap_p']:.2f} tail={r['R1_tail_window']['gap_p']:.2f} nontmpl={r['R2_nontemplate_tail']['gap_p']:.2f}")

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
