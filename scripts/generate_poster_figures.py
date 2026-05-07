"""Generate poster figures from phase2_ablation.pt results."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase2_ablation.pt"
OUTDIR = ROOT / "docs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
data = torch.load(RESULTS, map_location="cpu", weights_only=False)
cl = data["causal_labels"]

# ---------- colors & markers (red-orange palette) ----------
LANG_COLORS = {
    "en": "#8B1A1A",   # deep red
    "zh": "#CC5500",   # burnt orange
    "es": "#E8890C",   # amber
    "bn": "#D4534B",   # coral red
    "sw": "#A0522D",   # sienna
}
TAG_MARKERS = {
    "LANGUAGE": "o",
    "SHARED": "s",
    "REASONING": "^",
    "JUNK": "D",
}
TAG_LABELS = {
    "LANGUAGE": "Language-only",
    "SHARED": "Shared (lang + reas)",
    "REASONING": "Reasoning-only",
    "JUNK": "Junk",
}

# ---------- Figure 1: ppl-delta x acc-delta scatter ----------
fig, ax = plt.subplots(figsize=(8, 6))

# Collect data
for (lang, feat), info in cl.items():
    tag = info["tag"]
    acc_d = info["acc_delta"]
    ppl_d = info["ppl_delta"]
    base_ppl = info["baseline_ppl"]
    # Relative ppl increase, log-scaled
    rel_ppl = np.log10(max(1 + ppl_d / base_ppl, 1e-1))

    ax.scatter(
        rel_ppl, acc_d,
        c=LANG_COLORS[lang],
        marker=TAG_MARKERS[tag],
        s=120, edgecolors="white", linewidths=0.5,
        zorder=3,
    )

# Threshold lines
ppl_thresh_rel = np.log10(1 + 0.05)  # 5% relative increase
ax.axvline(ppl_thresh_rel, color="gray", ls="--", lw=1, alpha=0.6)
ax.axhline(0.04, color="gray", ls="--", lw=1, alpha=0.6)
ax.axhline(-0.04, color="gray", ls="--", lw=1, alpha=0.6)

# Quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
kw = dict(fontsize=9, alpha=0.35, ha="center", va="center", style="italic")
mid_x_left = (xlim[0] + ppl_thresh_rel) / 2
mid_x_right = (ppl_thresh_rel + xlim[1]) / 2
ax.text(mid_x_right, 0.15, "LANGUAGE", **kw)
ax.text(mid_x_right, -0.15, "SHARED", **kw)
ax.text(mid_x_left, -0.15, "REASONING", **kw)
ax.text(mid_x_left, 0.15, "JUNK", **kw)

ax.set_xlabel(r"$\log_{10}$(1 + $\Delta$ppl / baseline ppl)", fontsize=11)
ax.set_ylabel(r"$\Delta$accuracy (ablated $-$ baseline)", fontsize=11)
ax.set_title("Per-Feature Causal Profile (L17, top-5 A$\\cap$B per lang)", fontsize=12)
ax.axhline(0, color="black", lw=0.5, alpha=0.3)

# Language legend
from matplotlib.lines import Line2D
lang_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
           markersize=9, label=lang.upper())
    for lang, c in LANG_COLORS.items()
]
# Tag legend
tag_handles = [
    Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
           markersize=9, label=TAG_LABELS[t])
    for t, m in TAG_MARKERS.items()
]
leg1 = ax.legend(handles=lang_handles, loc="upper left", fontsize=8,
                 title="Language", title_fontsize=9, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=tag_handles, loc="lower right", fontsize=8,
          title="Tag", title_fontsize=9, framealpha=0.9)

ax.grid(True, alpha=0.15)
fig.tight_layout()
fig.savefig(OUTDIR / "ppl_acc_scatter.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "ppl_acc_scatter.png", dpi=200, bbox_inches="tight")
print(f"Saved scatter plot to {OUTDIR / 'ppl_acc_scatter.pdf'}")

# ---------- Figure 2: Jaccard overlap bar chart ----------
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

langs = ["en", "zh", "es", "bn", "sw"]
mgsm_only = [19-4, 6-5, 9-3, 16-7, 11-7]
shared = [4, 5, 3, 7, 7]
flores_only = [12-4, 9-5, 10-3, 11-7, 13-7]
jaccard = [0.148, 0.500, 0.188, 0.350, 0.412]

x = np.arange(len(langs))
w = 0.55

bars1 = ax2.bar(x, mgsm_only, w, label="MGSM-only", color="#8B1A1A")
bars2 = ax2.bar(x, shared, w, bottom=mgsm_only, label="Shared", color="#CC5500")
bars3 = ax2.bar(x, flores_only, w, bottom=[m+s for m,s in zip(mgsm_only, shared)],
                label="FLORES-only", color="#E8C47A")

# Jaccard annotation above each bar
for i, j in enumerate(jaccard):
    total = mgsm_only[i] + shared[i] + flores_only[i]
    ax2.text(x[i], total + 0.4, f"J={j:.2f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold",
             color="#8B1A1A" if j < 0.25 else "black")

ax2.set_xticks(x)
ax2.set_xticklabels([l.upper() for l in langs], fontsize=11)
ax2.set_ylabel("Number of features", fontsize=11)
ax2.set_title("MGSM vs FLORES Feature Overlap (L17)", fontsize=12)
ax2.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax2.grid(axis="y", alpha=0.15)
ax2.set_ylim(0, max(m+s+f for m,s,f in zip(mgsm_only, shared, flores_only)) + 3)

fig2.tight_layout()
fig2.savefig(OUTDIR / "jaccard_overlap.pdf", bbox_inches="tight")
fig2.savefig(OUTDIR / "jaccard_overlap.png", dpi=200, bbox_inches="tight")
print(f"Saved overlap chart to {OUTDIR / 'jaccard_overlap.pdf'}")

# ---------- Figure 3: Feature 96 dormant-to-active bar chart ----------
fig3, ax3 = plt.subplots(figsize=(7, 4))

langs_96 = ["SW", "ES", "EN", "ZH", "BN"]
clean_vals = [0, 0, 0, 0, 0]
ablated_vals = [7036, 2856, 2577, 2501, 2345]

x3 = np.arange(len(langs_96))
w3 = 0.35

ax3.bar(x3 - w3/2, clean_vals, w3, label="Clean (baseline)", color="#E8C47A",
        edgecolor="white")
ax3.bar(x3 + w3/2, ablated_vals, w3, label="After ablation", color="#8B1A1A",
        edgecolor="white")

for i, v in enumerate(ablated_vals):
    ax3.text(x3[i] + w3/2, v + 100, f"+{v}", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#8B1A1A")

ax3.set_xticks(x3)
ax3.set_xticklabels(langs_96, fontsize=11)
ax3.set_ylabel("Feature 96 activation", fontsize=11)
ax3.set_title("Dormant Reasoning Feature Released by Ablation (L17)", fontsize=12)
ax3.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax3.grid(axis="y", alpha=0.15)

fig3.tight_layout()
fig3.savefig(OUTDIR / "feat96_dormant.pdf", bbox_inches="tight")
fig3.savefig(OUTDIR / "feat96_dormant.png", dpi=200, bbox_inches="tight")
print(f"Saved feature 96 chart to {OUTDIR / 'feat96_dormant.pdf'}")

print("\nAll figures generated.")
