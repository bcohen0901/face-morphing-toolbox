"""Same-person benchmark from real CFD photographs.

For every CFD identity with a happy (closed-mouth, HC) photo, compares the
neutral photo with the HC photo of the same person, and with the neutral
photos of other people of the same race and gender. Writes benchmark.json.

    python benchmark.py
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def _env(name):
    """Path from an environment variable (see paper/README.md)."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set the {name} environment variable (see paper/README.md).")
    return value


CFD = _env("CFD_DIR")
HERE = os.path.dirname(os.path.abspath(__file__))

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))


def embed(path):
    """Unit-length ArcFace embedding of the largest face in the image at `path`."""
    faces = app.get(cv2.imread(path))
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


pairs = {}
for hc in sorted(glob.glob(f"{CFD}/Images/CFD/*/CFD-*-HC.jpg")):
    group, num = re.search(r"CFD-([A-Z]{2})-(\d+)-", os.path.basename(hc)).groups()
    neutral = glob.glob(f"{CFD}/CFD neutral (ALL)/CFD-{group}-{num}-*-N.jpg")
    if neutral:
        pairs[(group, num)] = (neutral[0], hc)

emb = {k: (embed(n), embed(h)) for k, (n, h) in pairs.items()}
by_group = defaultdict(list)
for group, num in emb:
    by_group[group].append(num)

same = [float(n @ h) for n, h in emb.values()]
diff = []
for group, nums in by_group.items():
    for a in range(len(nums)):
        for b in range(a + 1, len(nums)):
            diff.append(float(emb[(group, nums[a])][0] @ emb[(group, nums[b])][0]))

out = dict(
    n_identities=len(same),
    groups={g: len(v) for g, v in sorted(by_group.items())},
    same_person_mean=float(np.mean(same)),
    same_person_sd=float(np.std(same, ddof=1)),
    same_person_range=[float(min(same)), float(max(same))],
    different_person_mean=float(np.mean(diff)),
    different_person_sd=float(np.std(diff, ddof=1)),
    different_person_p99=float(np.percentile(diff, 99)),
    n_different_pairs=len(diff),
)
with open(os.path.join(HERE, "benchmark.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
