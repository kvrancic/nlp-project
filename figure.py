import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib as mpl

# ----------------------------
# Editable figure parameters
# ----------------------------
feature_id = "1105"

languages = ["EN", "FR", "ZH"]
activations = [0.18, 0.92, 0.26]   # illustrative values
highlight_lang = "FR"

baseline_output = "... The answer is 0.23"
ablated_output = "... The answer is 0.11"

out_base = "feature_1105_ablation_figure"

# ----------------------------
# Style
# ----------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

ink = "#1f1f1f"
muted = "#6f6f6f"
light = "#f7f7f7"
line = "#d9d9d9"

blue = "#3b6ea8"
blue_light = "#e8f0fa"

green = "#3f8f3f"
green_light = "#edf7ed"

red = "#b63a32"
red_light = "#faeeee"


# ----------------------------
# Helper drawing functions
# ----------------------------
def rounded_box(ax, xy, w, h, fc="white", ec=ink, lw=1.0, radius=0.08):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    return box


def draw_switch(ax, center, on=True, scale=1.0):
    """
    Draw a vertical toggle switch in axes coordinates.
    """
    cx, cy = center
    w = 0.16 * scale
    h = 0.42 * scale

    color = green if on else red
    fill = green_light if on else red_light

    rounded_box(
        ax,
        (cx - w / 2, cy - h / 2),
        w,
        h,
        fc=fill,
        ec=color,
        lw=1.5,
        radius=0.04 * scale,
    )

    knob_r = 0.045 * scale
    knob_y = cy + h * 0.25 if on else cy - h * 0.25

    ax.add_line(Line2D(
        [cx, cx],
        [cy - h * 0.24, cy + h * 0.24],
        transform=ax.transAxes,
        color=color,
        linewidth=2.5 * scale,
        solid_capstyle="round",
        clip_on=False,
    ))

    ax.add_patch(Circle(
        (cx, knob_y),
        knob_r,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="none",
        clip_on=False,
    ))


def output_box(ax, xy, text, color, fill, label):
    """
    Draw a small model-output card.
    """
    x, y = xy
    rounded_box(ax, (x, y), 0.78, 0.28, fc=fill, ec=color, lw=1.2, radius=0.045)

    ax.text(
        x + 0.035, y + 0.205,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        color=color,
        weight="bold",
    )

    ax.text(
        x + 0.035, y + 0.095,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color=ink,
    )


# ----------------------------
# Figure layout
# ----------------------------
fig = plt.figure(figsize=(7.4, 3.1), dpi=300)
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.05, 1.25, 1.85],
    left=0.045,
    right=0.985,
    bottom=0.18,
    top=0.90,
    wspace=0.42,
)

ax_model = fig.add_subplot(gs[0, 0])
ax_act = fig.add_subplot(gs[0, 1])
ax_ablate = fig.add_subplot(gs[0, 2])

# ----------------------------
# Panel A: LLM feature
# ----------------------------
ax_model.set_axis_off()

rounded_box(
    ax_model,
    (0.08, 0.18),
    0.78,
    0.64,
    fc=light,
    ec=ink,
    lw=1.0,
    radius=0.04,
)

ax_model.text(
    0.47, 0.60,
    "LLM",
    transform=ax_model.transAxes,
    ha="center",
    va="center",
    fontsize=18,
    weight="bold",
    color=ink,
)

rounded_box(
    ax_model,
    (0.22, 0.36),
    0.56,
    0.08,
    fc="white",
    ec=line,
    lw=1.0,
    radius=0.035,
)

ax_model.text(
    0.47, 0.40,
    f"feature {feature_id}",
    transform=ax_model.transAxes,
    ha="center",
    va="center",
    fontsize=7.5,
    color=ink,
)

ax_model.annotate(
    "",
    xy=(1.10, 0.50),
    xytext=(0.88, 0.50),
    xycoords=ax_model.transAxes,
    textcoords=ax_model.transAxes,
    arrowprops=dict(arrowstyle="->", color=muted, lw=1.2),
    clip_on=False,
)

# ----------------------------
# Panel B: language activation
# ----------------------------
bar_colors = [
    blue if lang == highlight_lang else "#c8c8c8"
    for lang in languages
]

ax_act.bar(languages, activations, color=bar_colors, width=0.58)

ax_act.set_ylim(0, 1.05)
ax_act.set_ylabel("Feature activation", fontsize=10.5)
ax_act.set_xlabel("Language", fontsize=10.5)

ax_act.spines["top"].set_visible(False)
ax_act.spines["right"].set_visible(False)
ax_act.spines["left"].set_color(muted)
ax_act.spines["bottom"].set_color(muted)

ax_act.tick_params(axis="both", labelsize=10, colors=ink)
ax_act.set_yticks([0, 0.5, 1.0])
ax_act.grid(axis="y", color="#ededed", linewidth=0.8)
ax_act.set_axisbelow(True)

# Annotate FR bar
fr_idx = languages.index(highlight_lang)
ax_act.text(
    fr_idx,
    activations[fr_idx] + 0.055,
    "selective",
    ha="center",
    va="bottom",
    fontsize=9.5,
    color=blue,
    weight="bold",
)

# ----------------------------
# Panel C: ablation and outputs
# ----------------------------
ax_ablate.set_axis_off()

ax_ablate.text(
    0.02, 0.88,
    "Ablating the feature changes the model output",
    transform=ax_ablate.transAxes,
    ha="left",
    va="center",
    fontsize=12.5,
    color=ink,
)

# On condition
draw_switch(ax_ablate, center=(0.12, 0.62), on=True, scale=1.0)
ax_ablate.text(
    0.23, 0.68,
    "feature active",
    transform=ax_ablate.transAxes,
    ha="left",
    va="center",
    fontsize=10.5,
    color=muted,
)
output_box(
    ax_ablate,
    xy=(0.23, 0.48),
    text=baseline_output,
    color=red,
    fill=red_light,
    label="original output",
)

# X mark
ax_ablate.text(
    0.86, 0.58,
    "✕",
    transform=ax_ablate.transAxes,
    ha="center",
    va="center",
    fontsize=20,
    color=red,
    weight="bold",
)

# Off condition
draw_switch(ax_ablate, center=(0.12, 0.25), on=False, scale=1.0)
ax_ablate.text(
    0.23, 0.31,
    "feature ablated",
    transform=ax_ablate.transAxes,
    ha="left",
    va="center",
    fontsize=10.5,
    color=muted,
)
output_box(
    ax_ablate,
    xy=(0.23, 0.11),
    text=ablated_output,
    color=green,
    fill=green_light,
    label="ablated output",
)

# Check mark
ax_ablate.text(
    0.86, 0.21,
    "✓",
    transform=ax_ablate.transAxes,
    ha="center",
    va="center",
    fontsize=21,
    color=green,
    weight="bold",
)

# Thin visual separator between activation and ablation panels
bbox = ax_ablate.get_position()
fig.add_artist(Line2D(
    [bbox.x0 - 0.025, bbox.x0 - 0.025],
    [bbox.y0 + 0.02, bbox.y1 - 0.02],
    transform=fig.transFigure,
    color=line,
    linewidth=0.8,
))

# ----------------------------
# Save
# ----------------------------
fig.savefig(f"{out_base}.png", dpi=300)
# fig.savefig(f"{out_base}.pdf")
# fig.savefig(f"{out_base}.svg")

# plt.show()