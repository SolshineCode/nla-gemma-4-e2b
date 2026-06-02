"""
LLM-judge forced-choice content-specificity eval (most sensitive test).

For each of the 50 v0.1 AV outputs, an LLM judge is shown the AV explanation plus
5 candidate source texts: the TRUE source document and 4 distractors drawn from
OTHER documents. The judge picks which source the explanation best describes.

  chance accuracy = 1/5 = 20%

If the AV output carries content signal an intelligent reader can extract, judge
accuracy >> 20% (binomial test). If it sits at ~20%, the content-blind verdict
holds even under the most generous, semantics-aware probe.

Judge = `claude -p` subprocess with Haiku (Claude Code credits, not API billing).
Prompt piped via stdin; cwd = home to avoid CLAUDE.md hijack. Raw responses saved.

Usage:
  python eval_content_specificity_judge.py --smoke   # 1 batch (10 trials)
  python eval_content_specificity_judge.py           # all 50
"""
import json
import os
import re
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(REPO, "data", "stage1", "rl.parquet")
AV = os.path.join(REPO, "results", "content_aware_eval",
                  "av_outputs_v0_1_dd_n50_from_neuronpedia.json")
RESDIR = os.path.join(REPO, "results", "content_aware_eval")
HOME = os.path.expanduser("~")
MODEL = "claude-haiku-4-5-20251001"
GEMINI_WRAPPER = os.path.join(HOME, ".claude", "skills", "gemini-collab",
                              "scripts", "gemini_client.py")
N_CHOICES = 5
SRC_CHARS = 320
RNG = np.random.RandomState(0)


def build_trials():
    df = pd.read_parquet(RL)
    av = json.load(open(AV, encoding="utf-8"))["our_outputs"]
    n = len(av)
    src = [df["detokenized_text_truncated"].iloc[i] for i in range(n)]
    doc = [df["doc_id"].iloc[i] for i in range(n)]
    trials = []
    for i in range(n):
        # distractors: rows from DIFFERENT documents
        pool = [j for j in range(n) if doc[j] != doc[i]]
        distractors = list(RNG.choice(pool, size=N_CHOICES - 1, replace=False))
        cands = distractors + [i]
        RNG.shuffle(cands)
        true_pos = cands.index(i) + 1  # 1-indexed
        trials.append({
            "trial_id": i,
            "av_output": av[i],
            "candidates": [src[c][-SRC_CHARS:] for c in cands],
            "candidate_rows": [int(c) for c in cands],
            "true_pos": true_pos,
            "true_doc": doc[i],
        })
    return trials


def make_prompt(batch):
    lines = [
        "You are evaluating a neural-network interpretation tool. For each ITEM you are",
        "given a DESCRIPTION (an automated explanation of what a hidden activation vector",
        "encodes at one position in some source document) and 5 candidate SOURCE texts.",
        "Exactly one candidate is the document the activation came from; the other 4 are",
        "from unrelated documents. Pick the candidate the DESCRIPTION best matches on ANY",
        "genuine connection: topic, named entities, the event itself, a person's",
        "personality or role, tone, register, or any other content link (it need not be",
        "the surface topic). You MUST choose one number 1-5 even if unsure.",
        "",
        'Return ONLY a JSON array, one object per item: [{"trial_id": N, "choice": K}, ...]',
        "",
    ]
    for t in batch:
        lines.append(f"=== ITEM trial_id={t['trial_id']} ===")
        lines.append(f"DESCRIPTION: {t['av_output'].strip()}")
        for k, c in enumerate(t["candidates"], 1):
            lines.append(f"SOURCE {k}: {c.strip()}")
        lines.append("")
    return "\n".join(lines)


def call_judge_claude(prompt, model=MODEL):
    claude = shutil.which("claude") or "claude"
    proc = subprocess.run(
        [claude, "-p", "--model", model],
        input=prompt, capture_output=True, text=True, cwd=HOME, timeout=300,
        encoding="utf-8", errors="replace",
    )
    return proc.stdout.strip()


def call_judge_gemini(prompt):
    # Antigravity (Gemini 3.5 Flash) via the gemini-collab wrapper; prompt passed as argv.
    proc = subprocess.run(
        [sys.executable, GEMINI_WRAPPER, "--prompt", prompt, "--timeout", "300"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=360,
    )
    return proc.stdout.strip()


def parse(resp):
    m = re.search(r"\[.*\]", resp, re.DOTALL)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
        return {int(o["trial_id"]): int(o["choice"]) for o in arr if "choice" in o}
    except Exception:
        return {}


def main():
    smoke = "--smoke" in sys.argv
    backend = "gemini" if "--backend" in sys.argv and sys.argv[sys.argv.index("--backend") + 1] == "gemini" else "claude"
    model_override = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else MODEL
    call_judge = call_judge_gemini if backend == "gemini" else (lambda p: call_judge_claude(p, model_override))
    judge_model = "Gemini 3.5 Flash (Antigravity)" if backend == "gemini" else model_override
    tag = backend if backend == "gemini" else ("claude_" + model_override.split("-")[1] if "-" in model_override else "claude")
    batch_size = 5 if backend == "gemini" else 10
    OUT = os.path.join(RESDIR, f"content_specificity_judge_{tag}.json")
    RAW = os.path.join(RESDIR, f"content_specificity_judge_{tag}_raw.jsonl")
    trials = build_trials()
    if smoke:
        trials = trials[:batch_size]
    open(RAW, "w").close()
    choices = {}
    for b in range(0, len(trials), batch_size):
        batch = trials[b:b + batch_size]
        prompt = make_prompt(batch)
        resp = call_judge(prompt)
        parsed = parse(resp)
        choices.update(parsed)
        with open(RAW, "a", encoding="utf-8") as f:
            f.write(json.dumps({"batch_start": b, "prompt_chars": len(prompt),
                                "response": resp, "parsed": parsed}) + "\n")
        print(f"batch {b}-{b+len(batch)}: parsed {len(parsed)}/{len(batch)}")

    scored = [(t["trial_id"], choices.get(t["trial_id"]), t["true_pos"])
              for t in trials if t["trial_id"] in choices]
    correct = sum(1 for _, ch, tp in scored if ch == tp)
    n = len(scored)
    acc = correct / n if n else 0.0
    # one-sided binomial test: is accuracy > chance (0.2)?
    p = stats.binomtest(correct, n, 1.0 / N_CHOICES, alternative="greater").pvalue if n else 1.0
    result = {
        "config": {"model": judge_model, "backend": backend, "n_choices": N_CHOICES,
                   "src_chars": SRC_CHARS, "n_trials": n, "smoke": smoke,
                   "av_file": os.path.basename(AV)},
        "correct": correct, "n": n, "accuracy": acc,
        "chance": 1.0 / N_CHOICES,
        "binomial_p_greater_than_chance": p,
        "verdict": ("content signal (p<0.05)" if p < 0.05 else "no signal above chance"),
        "per_trial": [{"trial_id": tid, "choice": ch, "true_pos": tp, "correct": ch == tp}
                      for tid, ch, tp in scored],
    }
    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\naccuracy {correct}/{n} = {acc:.3f} (chance {1/N_CHOICES:.3f})  "
          f"binom p={p:.4f}  -> {result['verdict']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
