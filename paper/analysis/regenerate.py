"""Regenerate the East Asian / White doppelgangers at several transform strengths.

Uses the Face Morphing Toolbox library (app/app.py, v1.2.0) with
the CFD category averages. Shape strength is 1 at every level; color and
texture strengths vary. Output: regenerated/<level>/<GROUP>_<NUM>_as{A,W}.jpg

    python regenerate.py
"""

import os
import sys
import types
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import cohort


def _env(name):
    """Path from an environment variable (see paper/README.md)."""
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Set the {name} environment variable (see paper/README.md).")
    return value


APP = str(Path(__file__).resolve().parents[2] / "app" / "app.py")
STIM = _env("DOPP_STIM")
AVG = f"{STIM}/Avg faces"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "regenerated")

# level name -> (color strength, texture strength)
LEVELS = {
    "c30_t20": (0.30, 0.20),
    "c50_t50": (0.50, 0.50),
    "c75_t75": (0.75, 0.75),
    "c100_t100": (1.00, 1.00),
}
OTHER = {"A": "W", "W": "A"}


def load_lib():
    src = open(APP).read()
    lib = types.ModuleType("toolbox")
    lib.__file__ = APP
    exec(compile(src[: src.find("st.set_page_config")], APP, "exec"), lib.__dict__)
    return lib


lib = load_lib()
TRIS = lib._load_builtin_tris()


def read(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
    pts, curves = lib.load_tem(path[:-4] + ".tem")
    return img, pts, curves


AVERAGES = {g: read(f"{AVG}/CFD-{g}-AVG.jpg") for g in ("AF", "AM", "WF", "WM")}


def jobs():
    for group, num in cohort.identities():
        orig = cohort.original(group, num)
        for level in LEVELS:
            for target in (group[0], OTHER[group[0]]):
                yield group, num, orig, level, target


def run(job):
    group, num, orig, level, target = job
    out = os.path.join(OUT, level, f"{group}_{num}_as{target}.jpg")
    if os.path.exists(out):
        return out
    color, texture = LEVELS[level]
    imgB, ptsB, _ = read(orig)
    imgA, ptsA, _ = AVERAGES[group]
    imgC, ptsC, curves = AVERAGES[target + group[1]]
    params = dict(
        shape=1.0,
        color=color,
        texture=texture,
        feather=5,
        blur_sigma=10,
        pad=100,
        mask_output=True,
        bg_color=(128, 128, 128),
        color_anchor="subject",
        texture_anchor="subject",
        chroma_boost=1.0,
        norm_indices=(0, 1, 96),
        align="umeyama",
        face_poly=lib.BUILTIN_FACE_POLY,
        use_yuv=False,
        color_space="rgb",
        edge_fill=lib.EDGE_FILL,
    )
    img, _ = lib.run_transform_single(imgA, ptsA, imgB, ptsB, imgC, ptsC, curves, TRIS, params)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


if __name__ == "__main__":
    todo = list(jobs())
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for i, _ in enumerate(pool.imap_unordered(run, todo, chunksize=4)):
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print(f"done: {len(todo)} images")
