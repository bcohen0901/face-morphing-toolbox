# Manuscript materials

Scripts that reproduce the figures and the embedding analyses in Cohen,
Lakshmi, and Correll (in revision). They import the transformation code from
`app/app.py`, so they use exactly the same computation as the Toolbox.

Face images are not included. The Chicago Face Database (CFD) must be obtained
from its authors at https://www.chicagofaces.org under its own terms of use.

## Setup

Python 3.11:

```bash
pip install -r paper/requirements.txt
```

The scripts read their input locations from environment variables.

| Variable | Contents |
|----------|----------|
| `DOPP_STIM` | Folder with one subfolder per CFD group (`AF`, `AM`, `WF`, `WM`) holding the neutral CFD photographs and their `.tem` templates, an `Avg faces` subfolder holding `CFD-AF-AVG.jpg`, `CFD-AM-AVG.jpg`, `CFD-WF-AVG.jpg`, and `CFD-WM-AVG.jpg` with their `.tem` files |
| `CFD_DIR` | The CFD 3.0 folder, with `CFD neutral (ALL)` (Figure 6 and the same-person benchmark) and `Images/CFD` (same-person benchmark) |
| `FAIRFACE_WEIGHTS` | Path to `res34_fair_align_multi_7_20190809.pt` from the FairFace repository (Kärkkäinen & Joo, 2021) |

The ArcFace analyses use the insightface `buffalo_l` models, which insightface
downloads to `~/.insightface` on first use.

## Figures 1–4 and 6

```bash
DOPP_STIM=/path/to/stimuli CFD_DIR=/path/to/cfd python paper/figures/make_figures.py
```

Writes `fig_examples.png`, `fig_equation.png`, `fig_steps.png`, and
`fig_continuum.png` (Figures 1–4), and `fig_trust.png` (Figure 6) with shape
strength 1 and color and texture strengths of .75. Figure 6 builds its averages
from the CFD photographs listed in `figures/trust_faces.txt`.

## Embedding analyses (Figure 5, Table 3)

Run from `paper/analysis`:

```bash
python regenerate.py                 # Toolbox doppelgangers at four strengths
python embed.py                      # ArcFace embeddings of those images and the originals
python benchmark.py                  # same-person and different-person reference
python analyze.py embeddings_levels.npz levels           # all strengths (Table 3)
python analyze.py embeddings_levels.npz primary c75_t75  # Figure 5
python levels_summary.py             # strength summary table and chart
python fairface_embed.py             # FairFace embeddings and race predictions
python fairface_check.py             # FairFace race-classification check
MODEL=fairface python analyze.py embeddings_fairface_levels.npz fairface_primary c75_t75
```

The 76 identities analyzed are listed in `analysis/identities.txt`. The `results`
folder holds the outputs used in the manuscript.

## Familiarity pilot

`pilot/pilot_trials.csv` holds the de-identified test trials for the 50 participants
who finished, one row per trial. It contains no platform IDs, demographics, or
free-text answers. `pilot/results_summary.txt` lists the condition means and the
mixed-model results reported in the manuscript.

| Column | Contents |
|--------|----------|
| `participant` | Random participant label |
| `list` | Counterbalancing list (1–4) |
| `identity` | CFD identity of the source face |
| `source_cat`, `shown_cat` | Source and apparent category (A, B, L, W) |
| `role` | `old` = studied face; `dopp` = cross-category doppelganger of a studied face; `new_sc`, `new_xc` = new face, same- or cross-category |
| `rating` | Familiarity, from 1 (not at all familiar) to 7 (extremely familiar) |
| `rt` | Response time in ms |
| `flagged` | 1 = missed an attention check or performed poorly on the filler task |

The stimuli are not included, because they are derived from CFD photographs.
