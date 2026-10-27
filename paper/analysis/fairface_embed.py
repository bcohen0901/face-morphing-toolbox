"""FairFace embeddings and race predictions for the doppelganger stimuli.

FairFace (Karkkainen & Joo, 2021) is a ResNet-34 trained to classify race
(7 classes), gender, and age. Its 512-d penultimate-layer features capture
demographic appearance rather than identity, so they complement ArcFace.
Faces are aligned from the insightface 5-point landmarks to a 224 px crop
with margin around the face (approximating FairFace's dlib alignment).

Writes, in the same format as embed.py (plus race_prob):
    embeddings_fairface_levels.npz   Toolbox regenerations + originals
    benchmark_fairface.json          same-person / different-person reference

    python fairface_embed.py
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision
from insightface.app import FaceAnalysis

import embed  # manifests


def _env(name):
    """Path from an environment variable (see paper/README.md)."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set the {name} environment variable (see paper/README.md).")
    return value


WEIGHTS = _env("FAIRFACE_WEIGHTS")
CFD = _env("CFD_DIR")
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

# ArcFace 5-point template (112 px), rescaled to 224 px and shrunk about the
# center to leave a margin around the face. SCALE was chosen on the original
# photographs only (not the doppelgangers): 0.92 classified 74/76 correctly.
ARC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float32,
)
SCALE = 0.92
TEMPLATE = (ARC * 2 - 112) * SCALE + 112

model = torchvision.models.resnet34()
model.fc = torch.nn.Linear(model.fc.in_features, 18)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
model.eval()
backbone = torch.nn.Sequential(*list(model.children())[:-1])
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

detector = FaceAnalysis(
    name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"]
)
detector.prepare(ctx_id=-1, det_size=(640, 640))


def features(path):
    img = cv2.imread(path)
    faces = detector.get(img)
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    M, _ = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), TEMPLATE)
    crop = cv2.warpAffine(img, M, (224, 224), borderValue=(0, 0, 0))
    x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        feat = backbone(x).flatten(1)
        race = torch.softmax(model.fc(feat)[0, :7], 0).numpy()
    feat = feat[0].numpy()
    return feat / np.linalg.norm(feat), race


def save(rows, name):
    out = [features(r[4]) for r in rows]
    cols = list(zip(*rows))
    np.savez(
        os.path.join(HERE, name),
        set=cols[0],
        group=cols[1],
        identity=cols[2],
        version=cols[3],
        path=cols[4],
        embedding=np.array([o[0] for o in out]),
        race_prob=np.array([o[1] for o in out]),
    )
    print(f"{name}: {len(rows)} images")


def benchmark():
    emb = {}
    for hc in sorted(glob.glob(f"{CFD}/Images/CFD/*/CFD-*-HC.jpg")):
        group, num = re.search(r"CFD-([A-Z]{2})-(\d+)-", os.path.basename(hc)).groups()
        neutral = glob.glob(f"{CFD}/CFD neutral (ALL)/CFD-{group}-{num}-*-N.jpg")
        if neutral:
            emb[(group, num)] = (features(neutral[0])[0], features(hc)[0])
    by_group = defaultdict(list)
    for group, num in emb:
        by_group[group].append(num)
    same = [float(n @ h) for n, h in emb.values()]
    diff = [
        float(emb[(g, a)][0] @ emb[(g, b)][0])
        for g, nums in by_group.items()
        for i, a in enumerate(nums)
        for b in nums[i + 1 :]
    ]
    out = dict(
        n_identities=len(same),
        same_person_mean=float(np.mean(same)),
        same_person_sd=float(np.std(same, ddof=1)),
        different_person_mean=float(np.mean(diff)),
        different_person_sd=float(np.std(diff, ddof=1)),
        different_person_p99=float(np.percentile(diff, 99)),
    )
    with open(os.path.join(HERE, "benchmark_fairface.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    save(embed.levels_manifest(), "embeddings_fairface_levels.npz")
    benchmark()
