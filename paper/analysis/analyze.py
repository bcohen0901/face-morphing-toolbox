"""Identity-preservation analysis of ArcFace embeddings (reads embeddings.npz).

For each source identity i (source category s, other category o, gender g):
  pair       cos(same-category doppelganger_i, cross-category doppelganger_i)
  unrel_same mean cos(doppelganger_i, doppelgangers of other same-gender
             identities that share its apparent category), averaged over
             both of identity i's doppelgangers
  unrel_cross mean cos(doppelganger_i, doppelgangers of other same-gender
             identities in the other apparent category)
  orig_own   cos(original photo_i, same-category doppelganger_i)
  orig_other mean cos(original photo_i, same-category doppelgangers of other
             same-gender identities from the same source category)
Also: rank-1 identification of each doppelganger's partner among all
same-gender doppelgangers of the partner's apparent category, and an
odd-one-out triad test (doppelganger_i, partner_i, unrelated same-category
face); the model's odd one out is the item least similar to the other two.

    python analyze.py embeddings_levels.npz levels            # all strength levels
    python analyze.py embeddings_levels.npz primary c75_t75   # one level only
    MODEL=fairface python analyze.py embeddings_fairface_levels.npz fairface_primary c75_t75
"""

import json
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "arcface")
MODEL_NAME = {"arcface": "ArcFace", "fairface": "FairFace"}[MODEL]
NPZ = sys.argv[1] if len(sys.argv) > 1 else "embeddings_levels.npz"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
z = np.load(os.path.join(HERE, NPZ))
E = z["embedding"]
recs = [
    dict(set=s, group=g, id=i, version=v, k=k)
    for k, (s, g, i, v) in enumerate(zip(z["set"], z["group"], z["identity"], z["version"]))
]

ONLY = sys.argv[3] if len(sys.argv) > 3 else None
SETS = {s: ("asA", "asW") for s in dict.fromkeys(z["set"]) if ONLY in (None, s)}
LABEL = {
    s: "Color/texture " + "{:.2f} / {:.2f}".format(*[int(x) / 100 for x in s[1:].split("_t")])
    for s in SETS
}


def cos(a, b):
    return float(E[a] @ E[b])


def analyze(set_name):
    cat_versions = SETS[set_name]
    ids = defaultdict(dict)  # identity -> version -> row index
    meta = {}
    for r in recs:
        if r["set"] == set_name:
            ids[r["id"]][r["version"]] = r["k"]
            meta[r["id"]] = r["group"]
    ids = {i: v for i, v in ids.items() if all(x in v for x in ("original",) + cat_versions)}
    # Same-category version is "as" + the group's first letter (asA, asW)
    src_cat = {i: f"as{meta[i][0]}" for i in ids}
    gender = {i: meta[i][1] for i in ids}

    per_id = []
    ranks, triads, n_cands = [], [], []
    for i in ids:
        s = src_cat[i]
        o = [v for v in cat_versions if v != s][0]
        S, X, Orig = ids[i][s], ids[i][o], ids[i]["original"]
        same_g = [j for j in ids if j != i and gender[j] == gender[i]]

        def unrel(k, cat):
            return [cos(k, ids[j][cat]) for j in same_g]

        pair = cos(S, X)
        unrel_same = np.mean(unrel(S, s) + unrel(X, o))
        unrel_cross = np.mean(unrel(S, o) + unrel(X, s))
        orig_own = cos(Orig, S)
        orig_other = np.mean([cos(Orig, ids[j][s]) for j in same_g if src_cat[j] == s])
        per_id.append(
            dict(
                id=i,
                group=meta[i],
                pair=pair,
                unrel_same=unrel_same,
                unrel_cross=unrel_cross,
                orig_own=orig_own,
                orig_other=orig_other,
            )
        )

        # Rank of the true partner among all same-gender doppelgangers in its category
        for probe, partner, cat in ((S, X, o), (X, S, s)):
            cands = [partner] + [ids[j][cat] for j in same_g]
            sims = [cos(probe, c) for c in cands]
            ranks.append(1 + sum(x > sims[0] for x in sims[1:]))
            n_cands.append(len(cands))

        # Odd-one-out triads: (D_i, partner_i, unrelated face sharing D_i's category)
        for D, P, cat in ((S, X, s), (X, S, o)):
            for j in same_g:
                U = ids[j][cat]
                sim = {"D": cos(P, U), "P": cos(D, U), "U": cos(D, P)}
                # sim[x] is the similarity of the pair that excludes x, so the
                # odd one out is the item excluded from the most similar pair
                triads.append(max(sim, key=sim.get) == "U")
    return per_id, np.array(ranks), n_cands, np.array(triads)


def describe(x, y, label):
    d = np.asarray(x) - np.asarray(y)
    t = stats.ttest_rel(x, y)
    w = stats.wilcoxon(x, y)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return dict(
        comparison=label,
        n=len(d),
        mean_x=float(np.mean(x)),
        mean_y=float(np.mean(y)),
        mean_diff=float(d.mean()),
        ci95=[float(ci[0]), float(ci[1])],
        dz=float(d.mean() / d.std(ddof=1)),
        t=float(t.statistic),
        p_t=float(t.pvalue),
        p_wilcoxon=float(w.pvalue),
        prop_positive=float((d > 0).mean()),
    )


with open(
    os.path.join(HERE, "benchmark.json" if MODEL == "arcface" else f"benchmark_{MODEL}.json")
) as f:
    bench = json.load(f)
results = {}
fig, axes = plt.subplots(1, len(SETS), figsize=(5.6 * len(SETS), 4.6), sharey=True, squeeze=False)
axes = axes[0]
for ax, set_name in zip(axes, SETS):
    per_id, ranks, n_cands, triads = analyze(set_name)
    col = {k: np.array([r[k] for r in per_id]) for k in per_id[0] if k not in ("id", "group")}
    res = {
        "n_identities": len(per_id),
        "groups": {
            g: sum(r["group"] == g for r in per_id) for g in sorted({r["group"] for r in per_id})
        },
        "pair_vs_unrelated_same_category": describe(
            col["pair"], col["unrel_same"], "pair vs unrelated same-category"
        ),
        "pair_vs_unrelated_cross_category": describe(
            col["pair"], col["unrel_cross"], "pair vs unrelated other-category"
        ),
        "original_vs_own_vs_other": describe(
            col["orig_own"], col["orig_other"], "original-own vs original-other"
        ),
        "pair_vs_original_own": describe(col["pair"], col["orig_own"], "pair vs original-own"),
        "rank1_accuracy": float((ranks == 1).mean()),
        "mean_rank": float(ranks.mean()),
        "candidates_per_probe_range": [int(min(n_cands)), int(max(n_cands))],
        "chance_rank1_mean": float(np.mean(1.0 / np.array(n_cands))),
        "pair_min": float(col["pair"].min()),
        "pair_max": float(col["pair"].max()),
        "unrel_same_max_identity_mean": float(col["unrel_same"].max()),
        "odd_one_out_correct": float(triads.mean()),
        "n_triads": int(triads.size),
        "auc_pair_vs_unrel_same": float(
            stats.mannwhitneyu(col["pair"], col["unrel_same"]).statistic / (len(col["pair"]) ** 2)
        ),
    }
    results[set_name] = res

    keys = ["pair", "unrel_same", "unrel_cross", "orig_own", "orig_other"]
    names = [
        "Doppelganger\npair",
        "Unrelated,\nsame\ncategory",
        "Unrelated,\nother\ncategory",
        "Original vs\nits own\nsame-cat.\ndopp.",
        "Original vs\nunrelated\nsame-cat.\ndopps.",
    ]
    xs = [0, 1, 2, 3.5, 4.5]
    rng = np.random.default_rng(0)
    for x, k in zip(xs, keys):
        vals = col[k]
        color = "#2a78d6" if k in ("pair", "orig_own") else "#8a8984"
        ax.scatter(
            x + rng.uniform(-0.18, 0.18, len(vals)),
            vals,
            s=9,
            color=color,
            alpha=0.55,
            linewidths=0,
        )
        ax.plot([x - 0.28, x + 0.28], [vals.mean()] * 2, color="#0b0b0b", lw=2)
    for y, txt in (
        (bench["same_person_mean"], "Same person, two photos (mean)"),
        (bench["different_person_p99"], "Different people (99th pct.)"),
    ):
        ax.axhline(y, color="#52514e", lw=0.8, ls="--", zorder=0)
        ax.text(-0.45, y + 0.012, txt, fontsize=6.5, color="#52514e", ha="left")
    ax.set_xticks(xs, names, fontsize=7.5)
    ax.axvline(2.75, color="#d6d5d0", lw=0.8, zorder=0)
    if len(SETS) > 1:
        ax.set_title(f"{LABEL[set_name]} (n = {len(per_id)} identities)", fontsize=10)
    ax.axhline(0, color="#d6d5d0", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel(f"Cosine similarity of {MODEL_NAME} embeddings")
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"embedding_similarity{SUFFIX}.png"), dpi=300)
with open(os.path.join(HERE, f"results{SUFFIX}.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
