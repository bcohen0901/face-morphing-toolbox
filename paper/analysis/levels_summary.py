"""Summarize how embedding similarity changes with transform strength.

Reads results_levels.json (Toolbox regenerations at each strength) and
benchmark.json, and writes levels_summary.png and levels_summary.csv.

    python levels_summary.py
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

HERE = os.path.dirname(os.path.abspath(__file__))
lev = json.load(open(os.path.join(HERE, "results_levels.json")))

ORDER = ["c30_t20", "c50_t50", "c75_t75", "c100_t100"]
XLAB = ["0.30 / 0.20", "0.50 / 0.50", "0.75 / 0.75", "1.00 / 1.00"]
SERIES = [  # (label, results key, field, color)
    ("Doppelganger pair", "pair_vs_unrelated_same_category", "mean_x", "#2a78d6"),
    (
        "Original vs own same-category dopp.",
        "original_vs_own_vs_other",
        "mean_x",
        "#eb6834",
    ),
    (
        "Unrelated, same category",
        "pair_vs_unrelated_same_category",
        "mean_y",
        "#1baf7a",
    ),
    (
        "Unrelated, other category",
        "pair_vs_unrelated_cross_category",
        "mean_y",
        "#eda100",
    ),
]


def row(r):
    """Summary values for one strength level, keyed by their table column names."""
    out = {lbl: r[key][field] for lbl, key, field, _ in SERIES}
    out["Pair minus unrelated same category"] = r["pair_vs_unrelated_same_category"][
        "mean_diff"
    ]
    out["Pair minus original-own"] = r["pair_vs_original_own"]["mean_diff"]
    out["Rank-1 identification"] = r["rank1_accuracy"]
    out["Odd-one-out correct"] = r["odd_one_out_correct"]
    out["Pair min"] = r["pair_min"]
    return out


rows = [(f"Toolbox, color/texture {x}", row(lev[k])) for k, x in zip(ORDER, XLAB)]
with open(os.path.join(HERE, "levels_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Stimuli"] + list(rows[0][1]))
    for name, r in rows:
        w.writerow([name] + [f"{v:.3f}" for v in r.values()])

fig, ax = plt.subplots(figsize=(6.4, 4.4))
xs = range(len(ORDER))
for lbl, key, field, color in SERIES:
    ys = [lev[k][key][field] for k in ORDER]
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, label=lbl)
ax.set_xticks(list(xs), XLAB, fontsize=8)
ax.set_xlabel("Color / texture strength (shape = 1)", fontsize=9)
ax.set_ylabel("Mean cosine similarity (ArcFace)", fontsize=9)
ax.set_ylim(0, 0.9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
bench = json.load(open(os.path.join(HERE, "benchmark.json")))
ax.axhline(bench["same_person_mean"], color="#52514e", lw=0.8, ls="--", zorder=0)
ax.text(
    3.2,
    bench["same_person_mean"] + 0.012,
    "Same person, two photos (mean)",
    fontsize=7,
    color="#52514e",
    ha="right",
)
ax.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.44))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "levels_summary.png"), dpi=300)
for name, r in rows:
    print(name, {k: round(v, 3) for k, v in r.items()})
