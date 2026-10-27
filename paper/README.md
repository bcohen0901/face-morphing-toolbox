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
| `DOPP_STIM` | Folder with one subfolder per CFD group (`AF`, `AM`, `WF`, `WM`) holding the neutral CFD photographs and their `.tem` templates, an `Avg faces` subfolder holding `CFD-AF-AVG.jpg`, `CFD-AM-AVG.jpg`, `CFD-WF-AVG.jpg`, and `CFD-WM-AVG.jpg` with their `.tem` files, and (for `embed.py` only) `AF_asA`, `AF_asW`, and so on with the Psychomorph doppelgangers |
| `CFD_DIR` | The CFD 3.0 folder, with `CFD neutral (ALL)` and `Images/CFD` (used for the same-person benchmark) |
| `FAIRFACE_WEIGHTS` | Path to `res34_fair_align_multi_7_20190809.pt` from the FairFace repository (Kärkkäinen & Joo, 2021) |

The ArcFace analyses use the insightface `buffalo_l` models, which insightface
downloads to `~/.insightface` on first use.

## Figures 1–4

```bash
DOPP_STIM=/path/to/stimuli python paper/figures/make_figures.py
```

Writes `fig_examples.png`, `fig_equation.png`, `fig_steps.png`, and
`fig_continuum.png` (Figures 1–4) with shape strength 1 and color and texture
strengths of .75.

## Embedding analyses (Figure 5, Table 4)

Run from `paper/analysis`:

```bash
python regenerate.py                 # Toolbox doppelgangers at four strengths
python embed.py --levels             # ArcFace embeddings of those images
python embed.py                      # ArcFace embeddings of the Psychomorph stimuli
python benchmark.py                  # same-person and different-person reference
python analyze.py                                        # Psychomorph stimuli
python analyze.py embeddings_levels.npz levels           # all strengths (Table 4)
python analyze.py embeddings_levels.npz primary c75_t75  # Figure 5
python levels_summary.py             # strength summary table and chart
python fairface_embed.py             # FairFace embeddings and race predictions
python fairface_check.py             # FairFace race-classification check
MODEL=fairface python analyze.py embeddings_fairface_levels.npz fairface_primary c75_t75
```

The `results` folder holds the outputs used in the manuscript.
