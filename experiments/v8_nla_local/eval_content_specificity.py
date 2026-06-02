"""
Content-specificity retrieval eval for the v0.1 AV.

Question under test (challenge to the "content-blind / template-clustered" claim):
    Does the v0.1 AV text output carry ANY recoverable content signal about its
    source document, independent of the AR?

The published "content-blind" verdict rests on AV->AR activation-reconstruction
discriminability (cross-row argmax = 2/50, ~chance). That tests whether the AR can
USE the AV output to pick the right activation. It does NOT test whether the AV's
TEXT carries source-content signal. This eval tests the AV text directly via
retrieval against the source documents, with a permutation null.

Data: the n=50 AV outputs already generated for the Neuronpedia head-to-head
(av_outputs_v0_1_dd_n50_from_neuronpedia.json), row-aligned to rl.parquet rows 0..49.
Those 50 rows come from 13 unique documents (~4 token-positions each), so the fair,
well-posed task is DOC-LEVEL retrieval (chance = 1/13); row-level (chance = 1/50)
is reported as a harder secondary metric.

H0: AV text has no content signal -> doc-level top-1 ~ 1/13; same-doc sim ~ diff-doc sim.
H1: AV text has content signal      -> doc-level top-1 >> 1/13 and same-doc > diff-doc
                                        at permutation p < 0.05.

Pure-CPU, reuses already-generated outputs. No GPU, no model re-run.
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
                   "content_specificity_retrieval.json")

RNG = np.random.RandomState(0)
N_PERM = 5000


def load():
    df = pd.read_parquet(RL)
    av = json.load(open(AV, encoding="utf-8"))["our_outputs"]
    n = len(av)
    src = [df["detokenized_text_truncated"].iloc[i] for i in range(n)]
    doc = [df["doc_id"].iloc[i] for i in range(n)]
    # sanity: row-alignment already verified externally (rl[i][:120]==basic_func[i])
    return av, src, doc, n


def sim_matrix(av, src, mode):
    """sim[i, j] = similarity(AV_output_i, source_j). Rows=AV, cols=source."""
    if mode == "tfidf_word":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                              stop_words="english")
    elif mode == "tfidf_char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                              sublinear_tf=True)
    else:
        raise ValueError(mode)
    vec.fit(src + av)
    A = vec.transform(av)
    S = vec.transform(src)
    return cosine_similarity(A, S)


def sim_matrix_semantic(av, src):
    """Semantic embeddings via sentence-transformers, if installed. Returns None if not."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None, None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ea = model.encode(av, normalize_embeddings=True, show_progress_bar=False)
    es = model.encode(src, normalize_embeddings=True, show_progress_bar=False)
    return ea @ es.T, "all-MiniLM-L6-v2"


def row_level(sim):
    n = sim.shape[0]
    ranks = np.empty(n, dtype=int)
    for i in range(n):
        order = np.argsort(-sim[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "chance_top1": 1.0 / n,
        "chance_mean_rank": (n + 1) / 2.0,
    }


def doc_level(sim, doc):
    """For each AV_i, score each doc d = max over that doc's source rows of sim[i, j].
    Predict argmax doc; correct if == doc[i]. Self row (j==i) is allowed (it is in its
    own doc's pool, which is correct). Chance top1 = 1/n_docs."""
    n = sim.shape[0]
    docs = sorted(set(doc))
    d_index = {d: k for k, d in enumerate(docs)}
    cols_by_doc = {d: [j for j in range(n) if doc[j] == d] for d in docs}
    nd = len(docs)
    top1 = 0
    top3 = 0
    ranks = []
    for i in range(n):
        scores = np.array([sim[i, cols_by_doc[d]].max() for d in docs])
        order = np.argsort(-scores)
        true_k = d_index[doc[i]]
        rank = int(np.where(order == true_k)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 3:
            top3 += 1
    return {
        "n_docs": nd,
        "top1": top1 / n,
        "top3": top3 / n,
        "mean_rank": float(np.mean(ranks)),
        "chance_top1": 1.0 / nd,
        "chance_mean_rank": (nd + 1) / 2.0,
    }


def diag_offdiag(sim, doc):
    """Same-doc (i!=j) vs different-doc mean similarity."""
    n = sim.shape[0]
    same, diff = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (same if doc[i] == doc[j] else diff).append(sim[i, j])
    same, diff = np.array(same), np.array(diff)
    return float(same.mean()), float(diff.mean()), float(same.mean() - diff.mean())


def permutation_test(sim, doc):
    """Null: randomly relabel which AV output belongs to which row (permute AV rows),
    preserving doc structure. Recompute doc-level top1 and same-minus-diff gap.
    Empirical p = P(null >= observed)."""
    obs_doc = doc_level(sim, doc)["top1"]
    _, _, obs_gap = diag_offdiag(sim, doc)
    null_top1 = np.empty(N_PERM)
    null_gap = np.empty(N_PERM)
    idx = np.arange(sim.shape[0])
    for m in range(N_PERM):
        p = RNG.permutation(idx)
        sp = sim[p]  # permute AV rows; columns (source/doc) stay fixed
        null_top1[m] = doc_level(sp, doc)["top1"]
        _, _, null_gap[m] = diag_offdiag(sp, doc)
    def pz(obs, null):
        p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
        z = float((obs - null.mean()) / (null.std() + 1e-12))
        return p, z, float(null.mean())
    p1, z1, m1 = pz(obs_doc, null_top1)
    p2, z2, m2 = pz(obs_gap, null_gap)
    return {
        "doc_top1": {"observed": obs_doc, "null_mean": m1, "p_value": p1, "z": z1},
        "same_minus_diff_gap": {"observed": obs_gap, "null_mean": m2, "p_value": p2, "z": z2},
    }


def run_mode(name, sim, doc):
    same, diff, gap = diag_offdiag(sim, doc)
    return {
        "method": name,
        "row_level": row_level(sim),
        "doc_level": doc_level(sim, doc),
        "same_doc_mean_sim": same,
        "diff_doc_mean_sim": diff,
        "same_minus_diff": gap,
        "permutation": permutation_test(sim, doc),
    }


def main():
    av, src, doc, n = load()
    docs = sorted(set(doc))
    results = {
        "config": {
            "n_rows": n,
            "n_unique_docs": len(docs),
            "av_file": os.path.basename(AV),
            "source_field": "detokenized_text_truncated (truncated at activation token)",
            "n_permutations": N_PERM,
            "template_collapse": "18/50 share the 'country-specific...' prefix",
        },
        "methods": {},
    }
    for name, mode in [("tfidf_word", "tfidf_word"), ("tfidf_char", "tfidf_char")]:
        sim = sim_matrix(av, src, mode)
        results["methods"][name] = run_mode(name, sim, doc)
        print(f"[{name}] doc top1={results['methods'][name]['doc_level']['top1']:.3f} "
              f"(chance {1/len(docs):.3f})  gap p="
              f"{results['methods'][name]['permutation']['same_minus_diff_gap']['p_value']:.4f}")

    sem_sim, sem_model = sim_matrix_semantic(av, src)
    if sem_sim is not None:
        results["methods"]["semantic"] = run_mode(f"semantic:{sem_model}", sem_sim, doc)
        print(f"[semantic] doc top1={results['methods']['semantic']['doc_level']['top1']:.3f} "
              f"gap p={results['methods']['semantic']['permutation']['same_minus_diff_gap']['p_value']:.4f}")
    else:
        results["config"]["semantic"] = "sentence-transformers not installed; skipped"
        print("[semantic] skipped (sentence-transformers not installed)")

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
