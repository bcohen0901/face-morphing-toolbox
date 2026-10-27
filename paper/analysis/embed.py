"""Compute ArcFace embeddings for the East Asian / White doppelganger set.

Uses the insightface buffalo_l models (RetinaFace detector + ArcFace
ResNet-50 trained on WebFace600K), run locally on CPU. Writes one row per
image to embeddings.npz: set, group, identity, version, path, embedding.

    python embed.py            # Psychomorph set -> embeddings.npz
    python embed.py --levels   # regenerated/<level>/ -> embeddings_levels.npz
"""

import glob
import os
import re
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def _env(name):
    """Path from an environment variable (see paper/README.md)."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set the {name} environment variable (see paper/README.md).")
    return value


STIM = _env("DOPP_STIM")
HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = "--levels" in sys.argv
OUT = os.path.join(HERE, "embeddings_levels.npz" if LEVELS else "embeddings.npz")


def asian_white_manifest():
    """East Asian / White set (Cohen et al., 2026): original, asA, asW."""
    rows = []
    for group in ("AF", "AM", "WF", "WM"):
        for orig in sorted(glob.glob(f"{STIM}/{group}/CFD-{group}-*-N.jpg")):
            num = re.search(rf"CFD-{group}-(\d+)-", orig).group(1)
            ident = f"{group}-{num}"
            files = {
                "original": orig,
                "asA": f"{STIM}/{group}_asA/{group}_{num}_asA.jpg",
                "asW": f"{STIM}/{group}_asW/{group}_{num}_asW.jpg",
            }
            if all(os.path.exists(f) for f in files.values()):
                rows += [("AW", group, ident, v, f) for v, f in files.items()]
    return rows


def levels_manifest():
    """Toolbox regenerations at each strength level, with the same originals."""
    rows = []
    root = os.path.join(HERE, "regenerated")
    for level in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        for set_, group, ident, version, path in asian_white_manifest():
            if version != "original":
                num = ident.split("-")[1]
                path = os.path.join(HERE, "regenerated", level, f"{group}_{num}_{version}.jpg")
            rows.append((level, group, ident, version, path))
    return rows


def main():
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rows = levels_manifest() if LEVELS else asian_white_manifest()
    keep, embs, missed = [], [], []
    for i, row in enumerate(rows):
        faces = app.get(cv2.imread(row[4]))
        if not faces:
            missed.append(row[4])
            continue
        # Largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        keep.append(row)
        embs.append(face.normed_embedding)
        if i % 100 == 0:
            print(f"{i}/{len(rows)}")
    cols = list(zip(*keep))
    np.savez(
        OUT,
        set=cols[0],
        group=cols[1],
        identity=cols[2],
        version=cols[3],
        path=cols[4],
        embedding=np.array(embs),
    )
    print(f"embedded {len(keep)} images; no face found in {len(missed)}")
    for m in missed:
        print("  missed:", m)


if __name__ == "__main__":
    main()
