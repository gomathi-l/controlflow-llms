import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

bins = ["S", "M", "L", "VL"]

data = {
    "Haiku": {
        0:  [.393, .080, .098, .600],
        5:  [.687, .431, .589, .800],
        15: [.709, .485, .635, .800]
    },
    "Sonnet": {
        0:  [.397, .089, .067, .667],
        5:  [.692, .478, .475, .800],
        15: [.702, .539, .602, .733]
    },
    "GPT-4.1-nano": {
        0:  [.368, .098, .128, .067],
        5:  [.624, .375, .424, .333],
        15: [.660, .442, .622, .600]
    }
}

shots       = [0, 5, 15]
shot_labels = {
    0: "$n=0$ (zero-shot)",
    5: "$n=5$ (5-shot)",
    15: "$n=15$ (15-shot)"
}

shot_colors = {
    0: "#d62728",
    5: "#2ca02c",
    15: "#1f77b4"
}

shot_marker = {
    0: "o",
    5: "s",
    15: "D"
}

fig, axes = plt.subplots(1, 3, figsize=(3.5, 2.4), sharey=True)

for ax, (model, model_data) in zip(axes, data.items()):
    for shot in shots:
        ax.plot(
            bins,
            model_data[shot],
            marker=shot_marker[shot],
            linestyle="-",
            color=shot_colors[shot],
            linewidth=1.4,
            markersize=4,
            label=shot_labels[shot],
        )

    ax.set_title(model, pad=2)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Accuracy")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=True,
    framealpha=0.95,
    edgecolor="0.8",
    bbox_to_anchor=(0.5, -0.08)
)

# Tight layout spacing
plt.subplots_adjust(
    wspace=0.18,
    bottom=0.22
)

plt.savefig("length_sepsis.pdf", bbox_inches="tight")
plt.show()