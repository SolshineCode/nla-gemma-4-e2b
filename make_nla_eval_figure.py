"""Regenerable evaluation figure for the Gemma-4-E2B NLA model cards.

Produces `figures/nla_eval_across_versions.png` — a cross-version view of the two evals that
actually bear on the released AV: content-fidelity doc-level retrieval (the faithfulness proxy),
and round-trip cosine (reconstruction; structural-projection-dominated, shown with that caveat).

UPDATE THIS when a new NLA version ships or a new evaluation lands: add/extend an entry in
AV_VERSIONS (and REFS if a new reference line is measured), then re-run:
    python make_nla_eval_figure.py
The model cards embed the PNG, so the figure stays in sync with the eval record.

Doc-level retrieval = "does the AV's text for an activation retrieve that activation's own source
document among the held-out set?" (top-1, semantic/MiniLM). Reported per evaluation DOMAIN: the
original eval used out-of-domain political-news text; the in-domain eval uses held-out web text of
the kind represented in training. Reference lines: random chance, and the activation CEILING (what
a probe recovers from the raw L23 activation — i.e. how much content is present to be surfaced).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "nla_eval_across_versions.png")

# ---- the eval record (edit here when versions / evals update) ----
AV_VERSIONS = {
    "v0.0.1": {
        "roundtrip_cos": 0.438, "roundtrip_err": 0.054,
        "doc_ood": None, "doc_indomain": None,   # content-specificity not yet run on v0.0.1
    },
    "v0.1": {
        "roundtrip_cos": 0.460, "roundtrip_err": 0.054,
        "doc_ood": 0.09,        # out-of-domain (political news), n=50; at chance
        "doc_indomain": 0.14,   # in-domain (held-out web text); UPDATE with the n=160 number when confirmed
    },
}
REFS = {
    "chance_doc": 0.077,            # 1/13 docs in the matched eval set
    "activation_ceiling_doc": 0.24, # raw-L23 activation doc-retrieval (content present to surface)
    "roundtrip_noise_floor": 0.30,
}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    versions = list(AV_VERSIONS)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel A: content-fidelity doc-level retrieval (the faithfulness proxy) ----
    have = [v for v in versions if AV_VERSIONS[v]["doc_indomain"] is not None]
    x = np.arange(len(have)); w = 0.36
    ood = [AV_VERSIONS[v]["doc_ood"] for v in have]
    ind = [AV_VERSIONS[v]["doc_indomain"] for v in have]
    axA.bar(x - w/2, ood, w, label="out-of-domain (news)", color="#bdbdbd", edgecolor="black")
    axA.bar(x + w/2, ind, w, label="in-domain (web)", color="#4c78a8", edgecolor="black")
    axA.axhline(REFS["chance_doc"], ls="--", color="black", lw=1,
                label=f"chance ({REFS['chance_doc']:.3f})")
    axA.axhline(REFS["activation_ceiling_doc"], ls=":", color="#d62728", lw=1.5,
                label=f"activation ceiling ({REFS['activation_ceiling_doc']:.2f})")
    axA.set_xticks(x); axA.set_xticklabels(have)
    axA.set_ylabel("doc-level retrieval (top-1)")
    axA.set_title("Content fidelity — verbalizer is domain-sensitive")
    axA.set_ylim(0, max(0.27, REFS["activation_ceiling_doc"] + 0.03))
    for xi, (o, i) in enumerate(zip(ood, ind)):
        axA.text(xi - w/2, o + 0.004, f"{o:.2f}", ha="center", va="bottom", fontsize=8)
        axA.text(xi + w/2, i + 0.004, f"{i:.2f}", ha="center", va="bottom", fontsize=8)
    axA.legend(fontsize=8, loc="upper left")

    # ---- Panel B: round-trip cosine across versions (reconstruction; caveated) ----
    xv = np.arange(len(versions))
    cos = [AV_VERSIONS[v]["roundtrip_cos"] for v in versions]
    err = [AV_VERSIONS[v]["roundtrip_err"] for v in versions]
    axB.bar(xv, cos, 0.5, yerr=err, capsize=4, color="#72b7b2", edgecolor="black")
    axB.axhline(REFS["roundtrip_noise_floor"], ls="--", color="black", lw=1,
                label=f"noise floor ({REFS['roundtrip_noise_floor']:.2f})")
    axB.set_xticks(xv); axB.set_xticklabels(versions)
    axB.set_ylabel("round-trip cosine")
    axB.set_title("Reconstruction cosine (structural-projection dominated)")
    axB.set_ylim(0, 0.6)
    for xi, (c, e) in enumerate(zip(cos, err)):
        axB.text(xi, c + e + 0.01, f"{c:.3f}", ha="center", va="bottom", fontsize=8)
    axB.legend(fontsize=8, loc="upper right")
    axB.text(0.5, -0.16, "Round-trip cosine is ~95% AR structural projection, not per-row "
             "faithfulness — read alongside Panel A, not alone.",
             transform=axB.transAxes, ha="center", fontsize=7.5, style="italic", color="#555")

    fig.suptitle("Gemma-4-E2B NLA AV — evaluation across released versions", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
