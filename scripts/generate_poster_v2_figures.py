"""Generate poster v2 figures."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "docs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

LANGS = ["EN", "ZH", "ES", "BN", "SW"]

# ---------- Figure: Ablation accuracy (2 bars: baseline vs SAE) ----------
fig, ax = plt.subplots(figsize=(7, 4.5))

baseline = [0.580, 0.624, 0.568, 0.564, 0.320]
sae      = [0.712, 0.596, 0.636, 0.480, 0.296]

x = np.arange(len(LANGS))
w = 0.32

bars1 = ax.bar(x - w/2, baseline, w, label="Baseline", color="#E8C47A",
               edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + w/2, sae, w, label="After SAE ablation", color="#8B1A1A",
               edgecolor="white", linewidth=0.5)

# Delta annotations
for i, (b, s) in enumerate(zip(baseline, sae)):
    delta = s - b
    color = "#27ae60" if delta > 0 else "#c0392b"
    sign = "+" if delta > 0 else ""
    y_pos = max(b, s) + 0.015
    ax.text(x[i], y_pos, f"{sign}{delta:.1%}".replace("%", "pp").replace("0.", "."),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

# Mean line
mean_base = np.mean(baseline)
mean_sae = np.mean(sae)
ax.axhline(mean_base, color="#E8C47A", ls="--", lw=1, alpha=0.6, zorder=0)
ax.axhline(mean_sae, color="#8B1A1A", ls="--", lw=1, alpha=0.6, zorder=0)
ax.text(len(LANGS) - 0.5, mean_base + 0.01, f"mean={mean_base:.3f}",
        fontsize=8, color="#A0522D", ha="right")
ax.text(len(LANGS) - 0.5, mean_sae - 0.025, f"mean={mean_sae:.3f}",
        fontsize=8, color="#8B1A1A", ha="right")

ax.set_xticks(x)
ax.set_xticklabels(LANGS, fontsize=11)
ax.set_ylabel("MGSM Accuracy", fontsize=11)
ax.set_title("SAE Ablation vs Baseline (L17, k=20)", fontsize=12)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.15)
ax.set_ylim(0, 0.82)

fig.tight_layout()
fig.savefig(OUTDIR / "ablation_bars.pdf", bbox_inches="tight")
fig.savefig(OUTDIR / "ablation_bars.png", dpi=200, bbox_inches="tight")
print(f"Saved ablation bar chart to {OUTDIR / 'ablation_bars.pdf'}")
