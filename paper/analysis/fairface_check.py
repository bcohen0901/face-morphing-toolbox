"""Manipulation check: FairFace race predictions for originals and doppelgangers.

For each stimulus set, reports the percentage of images classified as the
intended apparent race (East Asian or Southeast Asian for asA and East Asian
originals; White for asW and White originals) and the mean probability of
the intended race. Writes fairface_check.csv.

    python fairface_check.py
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RACES = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]
ASIAN = [RACES.index("East Asian"), RACES.index("Southeast Asian")]
WHITE = [RACES.index("White")]

rows = []
for npz in ("embeddings_fairface_levels.npz",):
    z = np.load(os.path.join(HERE, npz))
    for s in dict.fromkeys(z["set"]):
        for version in ("original", "asA", "asW"):
            for src in ("A", "W"):
                m = (
                    (z["set"] == s)
                    & (z["version"] == version)
                    & np.array([g[0] == src for g in z["group"]])
                )
                probs = z["race_prob"][m]
                target = {"asA": "A", "asW": "W", "original": src}[version]
                idx = ASIAN if target == "A" else WHITE
                pred = probs.argmax(1)
                rows.append(
                    dict(
                        set=s,
                        version=version,
                        source=src,
                        n=int(m.sum()),
                        intended="East Asian" if target == "A" else "White",
                        pct_intended=float(np.isin(pred, idx).mean()),
                        mean_p_intended=float(probs[:, idx].sum(1).mean()),
                    )
                )
with open(os.path.join(HERE, "fairface_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(
        f"{r['set']:10s} {r['version']:8s} source={r['source']} -> {r['intended']:10s} "
        f"{r['pct_intended']:.0%}  p={r['mean_p_intended']:.2f}  (n={r['n']})"
    )
