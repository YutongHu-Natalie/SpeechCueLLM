import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
conditions = ["No context",  "Window = 3", "Window = 12",]

gpt_data = {
    "GPT-4o mini": {
        "Valence":   {"ZS": [0.6024,  0.6132, 0.5795], "FS": [0.6311,  0.6329, 0.6193]},
        "Arousal":   {"ZS": [0.3393,  0.34810,0.3787], "FS": [0.3589, 0.3462, 0.3720]},
        "Dominance": {"ZS": [0.1458, 0.1555, 0.1487], "FS": [0.1235, 0.1340, 0.1504]},
    },
    "GPT-5 mini": {
        "Valence":   {"ZS": [0.6630, 0.6806, 0.6892], "FS": [0.6697, 0.6857, 0.6861]},
        "Arousal":   {"ZS": [0.3926, 0.3710, 0.4264], "FS": [0.3416, 0.3415, 0.4226]},
        "Dominance": {"ZS": [0.2990, 0.3343, 0.3396], "FS": [0.3046, 0.3333, 0.3424]},
    },
}

# baseline from Table 4.9, windows from Table 5.7
llama_data = {
    "Valence": {
        "LLaMA-2-7B":    [0.7672, 0.6291, 0.6647],
        "LLaMA-3.1-8B":  [0.7433, 0.7160, 0.7677],
        "LLaMA-3.3-70B": [0.7822, 0.4626, 0.5368],
    },
    "Arousal": {
        "LLaMA-2-7B":    [0.4406, 0.3412, 0.3033],
        "LLaMA-3.1-8B":  [0.4778, 0.3968, 0.4411],
        "LLaMA-3.3-70B": [0.4653, 0.2034, 0.3046],
    },
    "Dominance": {
        "LLaMA-2-7B":    [0.4388, 0.3166, 0.2662],
        "LLaMA-3.1-8B":  [0.4400, 0.3719, 0.4252],
        "LLaMA-3.3-70B": [0.4413, 0.2412, 0.3585],
    },
}

dims = ["Valence", "Arousal", "Dominance"]
gpt_models = ["GPT-4o mini", "GPT-5 mini"]
llama_models = ["LLaMA-2-7B", "LLaMA-3.1-8B", "LLaMA-3.3-70B"]
x = np.arange(len(conditions))

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

gpt_style = {
    "GPT-4o mini": {"ZS": ("#185FA5", "-",  "o"), "FS": ("#378ADD", "--", "s")},
    "GPT-5 mini":  {"ZS": ("#085041", "-",  "o"), "FS": ("#1D9E75", "--", "s")},
}

llama_style = {
    "LLaMA-2-7B":    ("#993C1D", "-", "o"),
    "LLaMA-3.1-8B":  ("#185FA5", "-", "s"),
    "LLaMA-3.3-70B": ("#085041", "-", "^"),
}

# ── Figure 1: GPT — 2 rows (models) × 3 cols (dims) ─────────────────────────
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
fig1.suptitle("Impact of Past VAD Context on CCC — GPT Models", fontsize=13, y=1.01)

for row, model in enumerate(gpt_models):
    for col, dim in enumerate(dims):
        ax = axes1[row, col]
        for split, (color, ls, marker) in gpt_style[model].items():
            ax.plot(x, gpt_data[model][dim][split],
                    color=color, linestyle=ls, marker=marker,
                    linewidth=1.8, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_title(f"{model} — {dim}")
        ax.set_ylabel("CCC")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

handles1 = []
for model in gpt_models:
    for split, (color, ls, marker) in gpt_style[model].items():
        handles1.append(plt.Line2D([0], [0], color=color, linestyle=ls,
                                   marker=marker, linewidth=1.8, markersize=6,
                                   label=f"{model} — {split}"))
fig1.legend(handles=handles1, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("past_vad_gpt.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved past_vad_gpt.png")

# ── Figure 2: LoRA LLaMA — 1 row × 3 cols (dims), 3 models per panel ────────
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
fig2.suptitle("Impact of Past VAD Context on CCC — LoRA Fine-tuned LLaMA Models",
              fontsize=13, y=1.03)

for col, dim in enumerate(dims):
    ax = axes2[col]
    for model, (color, ls, marker) in llama_style.items():
        ax.plot(x, llama_data[dim][model],
                color=color, linestyle=ls, marker=marker,
                linewidth=1.8, markersize=7, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_title(dim)
    ax.set_ylabel("CCC")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

handles2 = [
    plt.Line2D([0], [0], color=color, linestyle=ls, marker=marker,
               linewidth=1.8, markersize=6, label=model)
    for model, (color, ls, marker) in llama_style.items()
]
fig2.legend(handles=handles2, loc="lower center", ncol=3,
            bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("past_vad_llama.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved past_vad_llama.png")