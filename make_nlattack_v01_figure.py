"""Regenerable NLAttack capability-floor figure for the v0.1 NLA model cards.
Reads the NLAttack emergence-dashboard result (9-axis EmergenceIndex with per-axis null margins)
and plots the available axes. Re-run after a fresh NLAttack run to update.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "experiments/v8_nla_local/results/nlattack/emergence_v01.json")))
ax_all = d["axes"]
avail = [a for a in ax_all if a.get("available") and a.get("score") == a.get("score")]  # drop nan
names = [a["name"] for a in avail]
scores = [a["score"] for a in avail]
margins = [max(a.get("margin", 0), 0) for a in avail]
na = [a["name"] for a in ax_all if a not in avail]
fig, ax = plt.subplots(figsize=(10, 5.5))
y = np.arange(len(names))
colors = ["#4c78a8" if s >= 0.5 else "#e0a458" for s in scores]
ax.barh(y, scores, color=colors, edgecolor="black")
ax.axvline(0.5, ls=":", color="gray", lw=1)
ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlim(0, 1.05); ax.set_xlabel("axis score (0-1, each null-margin-guarded)")
ax.set_title(f"NLAttack emergence dashboard — v0.1 NLA bottleneck\n"
             f"EmergenceIndex = {d['index']:.3f}  ·  \"{d['label']}\"")
for yi, s in zip(y, scores):
    ax.text(s + 0.01, yi, f"{s:.2f}", va="center", fontsize=9)
foot = ("decodability at ceiling = the activation carries the content (the verbalizer is the gap). "
        f"Axes needing extra setup (N/A here): {', '.join(na)}.")
fig.text(0.5, 0.015, foot, ha="center", fontsize=7.5, style="italic", color="#555", wrap=True)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(os.path.join(HERE, "figures/nlattack_v01_capability_floor.png"), dpi=130, bbox_inches="tight")
print("wrote figures/nlattack_v01_capability_floor.png | index", round(d["index"],3), "| axes", names)
