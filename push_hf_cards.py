"""Upload the updated model card(s) + the cross-version evaluation figure to the HuggingFace AV
repos. Run after the GitHub commit, when publishing the eval-accuracy update.

Token is read from the HF_TOKEN environment variable (NEVER hardcode it — this file lives in a
public repo). Usage:
    HF_TOKEN=hf_xxx python push_hf_cards.py

What it does:
  - uploads figures/nla_eval_across_versions.png to BOTH AV repos (cross-version eval viz),
  - uploads the updated MODEL_CARD_AV.md as README.md to the v0.1 repo (av-v0_1_dd-step_250) —
    that card is the integrated v0.1 card carrying the in-domain eval finding + the figure,
  - for the v0.0.1 repo, inserts the figure into its existing README (after the YAML frontmatter)
    without clobbering its v0.0.1-specific text.
"""
import os, sys
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    sys.exit("ERROR: set HF_TOKEN env var (do not hardcode the token in this public-repo file).")

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "nla_eval_across_versions.png")
CARD = os.path.join(HERE, "MODEL_CARD_AV.md")
V01_REPO = "Solshine/gemma-4-e2b-nla-L23-av-v0_1_dd-step_250"
V001_REPO = "Solshine/gemma-4-e2b-nla-L23-av-v0_0_1"
FIG_RELPATH = "figures/nla_eval_across_versions.png"
FIG_BLOCK = ("\n## Evaluation across released versions\n\n"
             f"![NLA AV evaluation across released versions]({FIG_RELPATH})\n\n"
             "Content-fidelity doc-level retrieval and reconstruction round-trip cosine across the "
             "released AV versions. The verbalizer's content-surfacing is domain-sensitive: at "
             "chance on out-of-domain news, modestly but significantly above chance in-domain. "
             "Round-trip cosine is structural-projection dominated, not a faithfulness metric. "
             "Regenerate with `make_nla_eval_figure.py` as new versions / evaluations land.\n")

api = HfApi(token=TOKEN)


def upload(path, path_in_repo, repo_id):
    url = api.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo, repo_id=repo_id,
                          repo_type="model", commit_message="Add cross-version evaluation figure + "
                          "domain-sensitive in-domain eval finding")
    print(f"  uploaded {path_in_repo} -> {repo_id}: {url}")


def main():
    # figure to both repos
    for repo in (V01_REPO, V001_REPO):
        upload(FIG, FIG_RELPATH, repo)

    # v0.1 repo: the integrated MODEL_CARD_AV.md becomes the README
    upload(CARD, "README.md", V01_REPO)

    # v0.0.1 repo: keep its text, just insert the figure block after the YAML frontmatter
    try:
        cur = hf_hub_download(repo_id=V001_REPO, filename="README.md", repo_type="model", token=TOKEN)
        txt = open(cur, encoding="utf-8").read()
        if FIG_RELPATH in txt:
            print(f"  {V001_REPO} README already has the figure; skipping text edit")
        else:
            if txt.startswith("---"):
                end = txt.find("\n---", 3)
                ins = end + len("\n---") if end != -1 else 0
                new = txt[:ins] + "\n" + FIG_BLOCK + txt[ins:]
            else:
                new = FIG_BLOCK + "\n" + txt
            tmp = os.path.join(HERE, "_v001_readme_tmp.md")
            open(tmp, "w", encoding="utf-8").write(new)
            upload(tmp, "README.md", V001_REPO)
            os.remove(tmp)
    except Exception as e:
        print(f"  WARN: could not update {V001_REPO} README text ({e}); figure still uploaded")

    print("HF cards updated.")


if __name__ == "__main__":
    main()
