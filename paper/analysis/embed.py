"""Compute ArcFace embeddings for the regenerated doppelgangers.

Uses the insightface buffalo_l models (RetinaFace detector + ArcFace
ResNet-50 trained on WebFace600K), run locally on CPU. For every identity in
identities.txt and every strength level in regenerated/, embeds the original
CFD photograph and both doppelgangers. Writes one row per image to
embeddings_levels.npz: set (the level), group, identity, version, path, embedding.

    python embed.py
"""

import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import cohort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "embeddings_levels.npz")


def levels_manifest():
    """Toolbox regenerations at each strength level, with the same originals."""
    rows = []
    root = os.path.join(HERE, "regenerated")
    for level in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        for group, num in cohort.identities():
            ident = f"{group}-{num}"
            rows.append((level, group, ident, "original", cohort.original(group, num)))
            for version in ("asA", "asW"):
                path = os.path.join(root, level, f"{group}_{num}_{version}.jpg")
                rows.append((level, group, ident, version, path))
    return rows


def main():
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    rows = levels_manifest()
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
