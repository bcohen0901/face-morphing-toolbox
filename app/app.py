"""
Face Morphing Toolbox — Streamlit App (Cloud Edition)
=====================================================
Seven tools for face morphing, transforming, and stimulus generation.
All inputs via browser upload, all outputs via zip download.
No local file paths needed — deployable on Streamlit Community Cloud.

Run locally:  streamlit run face_morphing_app_n.py
Requires:     pip install streamlit opencv-python-headless numpy

Landmark-alignment cutover, 2026-10-04
--------------------------------------
Anchor alignment (normalize_to_reference, used by morph_controlled and
average_faces) now defaults to a closed-form least-squares Umeyama fit.
It previously used cv2.estimateAffinePartial2D(..., LMEDS), which with only
three anchors fits a two-point subset exactly, abandons the third, and is not
translation-invariant.

Stimuli generated BEFORE this date used LMEDS. Nothing already on disk has
changed, but regenerating an older set now produces slightly different output
(a few px of landmark shift after cropping; ~2.5% mean pixel difference at
full strength). To rebuild a pre-cutover set byte-for-byte, pass
align="lmeds" — or set "align": "lmeds" in a pipeline config recipe.
"""

import base64
import csv
import datetime
import hashlib
import glob
import io
import json
import os
import re
import zipfile

import cv2
import numpy as np
import streamlit as st

APP_VERSION = "1.2.0"
EDGE_FILL = 0.01  # interior-fill band, as a fraction of face width (v1.2.0)

try:
    with open(__file__, "rb") as _f:
        APP_SOURCE_SHA256 = hashlib.sha256(_f.read()).hexdigest()
except (NameError, OSError):
    APP_SOURCE_SHA256 = None

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  TEM I/O                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def _parse_tem_lines(lines, expected_landmarks=189):
    """Shared parser for .tem content already split into lines."""
    start = 0
    try:
        int(lines[0].strip())
        start = 1
    except (ValueError, IndexError):
        pass
    pts, last_coord_line = [], start
    for i, line in enumerate(lines[start:], start=start):
        stripped = line.strip().replace(",", " ")
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            pts.append([float(parts[0]), float(parts[1])])
            last_coord_line = i
        except ValueError:
            continue
        if len(pts) >= expected_landmarks:
            break
    if len(pts) != expected_landmarks:
        raise ValueError(f"Expected {expected_landmarks} landmarks, got {len(pts)}")
    return np.array(pts, dtype=np.float32), lines[last_coord_line + 1:]


def load_tem(path, expected_landmarks=189):
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            if sum(1 for l in lines if len(l.strip().split()) >= 2) > 10:
                break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Could not read {path} with any known encoding.")
    return _parse_tem_lines(lines, expected_landmarks)


def load_tem_from_bytes(data: bytes, expected_landmarks=189):
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252"):
        try:
            text = data.decode(enc)
            if sum(1 for l in text.splitlines() if len(l.strip().split()) >= 2) > 10:
                break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not decode .tem with any known encoding.")
    return _parse_tem_lines(text.splitlines(keepends=True), expected_landmarks)


def save_tem(path, pts, curve_block=None):
    with open(path, "w") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts:
            f.write(f"{x:.4f}\t{y:.4f}\n")
        if curve_block:
            f.writelines(curve_block)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  MASK (.msk) AND SYMMETRY (.sym)                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

BUILTIN_FACE_POLY = [
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    112, 187, 113, 188, 114,
    133, 132, 131, 130, 129, 128, 127, 126, 125,
    111, 186, 110, 185,
]

LIP_SEAM_INDICES = list(range(95, 103))

# Built-in canonical triangulation (359 × 3, int64) — base64-encoded .npy
_BUILTIN_TRIS_B64 = (
    "k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBl"
    "JzogKDM1OSwgMyksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgIAoYAAAAAAAAACgAAAAAAAAACwAAAAAAAAAoAAAAAAAAABgAAAAAAAAAJwAA"
    "AAAAAAAAAAAAAAAAAAUAAAAAAAAABAAAAAAAAAAFAAAAAAAAAAAAAAAAAAAABgAAAAAAAAACAAAA"
    "AAAAAAMAAAAAAAAAFAAAAAAAAAADAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAABAAAAAAAAAA8AAAAA"
    "AAAADgAAAAAAAAAPAAAAAAAAAAEAAAAAAAAAEAAAAAAAAAABAAAAAAAAAA4AAAAAAAAADQAAAAAA"
    "AAAIAAAAAAAAAAAAAAAAAAAACQAAAAAAAAAAAAAAAAAAAAgAAAAAAAAABwAAAAAAAAADAAAAAAAA"
    "ACMAAAAAAAAAJAAAAAAAAAAjAAAAAAAAAAMAAAAAAAAAEwAAAAAAAAAUAAAAAAAAAAkAAAAAAAAA"
    "AgAAAAAAAAAJAAAAAAAAABQAAAAAAAAAJQAAAAAAAAAAAAAAAAAAAAIAAAAAAAAACQAAAAAAAAAU"
    "AAAAAAAAAAMAAAAAAAAAJAAAAAAAAAADAAAAAAAAAAAAAAAAAAAABAAAAAAAAAADAAAAAAAAAAQA"
    "AAAAAAAAEwAAAAAAAAAEAAAAAAAAABwAAAAAAAAAEgAAAAAAAAAcAAAAAAAAAAQAAAAAAAAABQAA"
    "AAAAAAATAAAAAAAAAAQAAAAAAAAAEgAAAAAAAAAGAAAAAAAAAAAAAAAAAAAABwAAAAAAAAAcAAAA"
    "AAAAAAUAAAAAAAAAHQAAAAAAAAAFAAAAAAAAAAYAAAAAAAAAHQAAAAAAAAAMAAAAAAAAABgAAAAA"
    "AAAACwAAAAAAAAAYAAAAAAAAAAwAAAAAAAAAHwAAAAAAAAAcAAAAAAAAAB0AAAAAAAAALAAAAAAA"
    "AAAGAAAAAAAAAAcAAAAAAAAAHQAAAAAAAAAMAAAAAAAAAAsAAAAAAAAAAQAAAAAAAAAHAAAAAAAA"
    "AAgAAAAAAAAAHgAAAAAAAAAeAAAAAAAAABUAAAAAAAAAFgAAAAAAAAAVAAAAAAAAAB4AAAAAAAAA"
    "CAAAAAAAAAAIAAAAAAAAAAkAAAAAAAAAFQAAAAAAAAAlAAAAAAAAABQAAAAAAAAAJAAAAAAAAAAB"
    "AAAAAAAAAAoAAAAAAAAAEQAAAAAAAAAKAAAAAAAAAAEAAAAAAAAACwAAAAAAAAAJAAAAAAAAACUA"
    "AAAAAAAAFQAAAAAAAAARAAAAAAAAABkAAAAAAAAAKQAAAAAAAAAZAAAAAAAAABEAAAAAAAAACgAA"
    "AAAAAAAoAAAAAAAAACcAAAAAAAAATQAAAAAAAAAKAAAAAAAAAAsAAAAAAAAAGQAAAAAAAAAdAAAA"
    "AAAAAAcAAAAAAAAAHgAAAAAAAAAMAAAAAAAAAAEAAAAAAAAADQAAAAAAAAAtAAAAAAAAAB0AAAAA"
    "AAAAHgAAAAAAAAAdAAAAAAAAAC0AAAAAAAAALAAAAAAAAAAMAAAAAAAAAA0AAAAAAAAAHwAAAAAA"
    "AAAvAAAAAAAAAB8AAAAAAAAAMAAAAAAAAAAfAAAAAAAAAC8AAAAAAAAAFwAAAAAAAAANAAAAAAAA"
    "AA4AAAAAAAAAIAAAAAAAAAAPAAAAAAAAACAAAAAAAAAADgAAAAAAAAAgAAAAAAAAAA8AAAAAAAAA"
    "IQAAAAAAAAAQAAAAAAAAAAEAAAAAAAAAEQAAAAAAAAChAAAAAAAAAEMAAAAAAAAAqQAAAAAAAABD"
    "AAAAAAAAAKEAAAAAAAAARAAAAAAAAAAPAAAAAAAAABAAAAAAAAAAIQAAAAAAAAAQAAAAAAAAABsA"
    "AAAAAAAAIQAAAAAAAAAbAAAAAAAAABAAAAAAAAAAGgAAAAAAAAAQAAAAAAAAABEAAAAAAAAAGgAA"
    "AAAAAAARAAAAAAAAACkAAAAAAAAAKgAAAAAAAAAiAAAAAAAAALkAAAAAAAAAbQAAAAAAAAC5AAAA"
    "AAAAACIAAAAAAAAAEgAAAAAAAAAiAAAAAAAAAG0AAAAAAAAARwAAAAAAAAATAAAAAAAAABIAAAAA"
    "AAAAIwAAAAAAAAAZAAAAAAAAAAsAAAAAAAAAKAAAAAAAAAAnAAAAAAAAABcAAAAAAAAAOAAAAAAA"
    "AAAXAAAAAAAAACcAAAAAAAAAGAAAAAAAAAAWAAAAAAAAABUAAAAAAAAAJgAAAAAAAAAfAAAAAAAA"
    "AA0AAAAAAAAAIAAAAAAAAAAyAAAAAAAAABYAAAAAAAAAJgAAAAAAAAAWAAAAAAAAADIAAAAAAAAA"
    "MwAAAAAAAAAYAAAAAAAAAB8AAAAAAAAAFwAAAAAAAAAmAAAAAAAAABUAAAAAAAAAJQAAAAAAAAAp"
    "AAAAAAAAABkAAAAAAAAAKAAAAAAAAAARAAAAAAAAACoAAAAAAAAAGgAAAAAAAAAaAAAAAAAAACoA"
    "AAAAAAAAGwAAAAAAAAArAAAAAAAAAHAAAAAAAAAAuwAAAAAAAABwAAAAAAAAACsAAAAAAAAAUgAA"
    "AAAAAAASAAAAAAAAABwAAAAAAAAApAAAAAAAAAASAAAAAAAAAKQAAAAAAAAAuQAAAAAAAAAcAAAA"
    "AAAAACwAAAAAAAAApAAAAAAAAAAfAAAAAAAAACAAAAAAAAAAMAAAAAAAAAAeAAAAAAAAABYAAAAA"
    "AAAALgAAAAAAAAAXAAAAAAAAAC8AAAAAAAAAOQAAAAAAAAAtAAAAAAAAAB4AAAAAAAAALgAAAAAA"
    "AAA1AAAAAAAAAC0AAAAAAAAALgAAAAAAAAAtAAAAAAAAADUAAAAAAAAAPQAAAAAAAAAgAAAAAAAA"
    "ACEAAAAAAAAAMQAAAAAAAAAhAAAAAAAAABsAAAAAAAAApwAAAAAAAAAxAAAAAAAAACEAAAAAAAAA"
    "pwAAAAAAAABHAAAAAAAAAFMAAAAAAAAAIgAAAAAAAABTAAAAAAAAAEcAAAAAAAAASAAAAAAAAAAj"
    "AAAAAAAAABIAAAAAAAAAIgAAAAAAAAAjAAAAAAAAACIAAAAAAAAAUwAAAAAAAABUAAAAAAAAAEkA"
    "AAAAAAAASgAAAAAAAABJAAAAAAAAAFQAAAAAAAAAUwAAAAAAAAAkAAAAAAAAACMAAAAAAAAAUwAA"
    "AAAAAAAkAAAAAAAAAFMAAAAAAAAAVAAAAAAAAAAlAAAAAAAAACQAAAAAAAAAVAAAAAAAAAAXAAAA"
    "AAAAADkAAAAAAAAAOAAAAAAAAAAmAAAAAAAAACUAAAAAAAAATAAAAAAAAAAyAAAAAAAAACYAAAAA"
    "AAAATAAAAAAAAAAlAAAAAAAAAEsAAAAAAAAATAAAAAAAAABLAAAAAAAAACUAAAAAAAAAVAAAAAAA"
    "AAApAAAAAAAAACgAAAAAAAAAVQAAAAAAAABKAAAAAAAAAIoAAAAAAAAASwAAAAAAAACKAAAAAAAA"
    "AEoAAAAAAAAAiQAAAAAAAABLAAAAAAAAAE4AAAAAAAAAMgAAAAAAAABOAAAAAAAAAEsAAAAAAAAA"
    "iwAAAAAAAAAbAAAAAAAAACoAAAAAAAAAKwAAAAAAAAAqAAAAAAAAACkAAAAAAAAAVgAAAAAAAABW"
    "AAAAAAAAACkAAAAAAAAAVQAAAAAAAAAbAAAAAAAAACsAAAAAAAAAuwAAAAAAAAArAAAAAAAAACoA"
    "AAAAAAAAVgAAAAAAAAAsAAAAAAAAAKUAAAAAAAAApAAAAAAAAAClAAAAAAAAACwAAAAAAAAApgAA"
    "AAAAAAAwAAAAAAAAACAAAAAAAAAAMQAAAAAAAAAsAAAAAAAAAC0AAAAAAAAAPQAAAAAAAAA0AAAA"
    "AAAAADMAAAAAAAAAOQAAAAAAAAAzAAAAAAAAADQAAAAAAAAALgAAAAAAAAAuAAAAAAAAABYAAAAA"
    "AAAAMwAAAAAAAAAuAAAAAAAAADQAAAAAAAAANQAAAAAAAAA1AAAAAAAAADQAAAAAAAAAOwAAAAAA"
    "AAAvAAAAAAAAADAAAAAAAAAAOwAAAAAAAAA7AAAAAAAAADoAAAAAAAAALwAAAAAAAAA6AAAAAAAA"
    "ADsAAAAAAAAANAAAAAAAAAAwAAAAAAAAADEAAAAAAAAAQgAAAAAAAACnAAAAAAAAABsAAAAAAAAA"
    "uwAAAAAAAACpAAAAAAAAAEIAAAAAAAAAMQAAAAAAAABCAAAAAAAAAKkAAAAAAAAAQwAAAAAAAABL"
    "AAAAAAAAAFQAAAAAAAAASgAAAAAAAAAzAAAAAAAAADgAAAAAAAAAOQAAAAAAAAA4AAAAAAAAADMA"
    "AAAAAAAAMgAAAAAAAAA0AAAAAAAAADkAAAAAAAAAOgAAAAAAAAA6AAAAAAAAADkAAAAAAAAALwAA"
    "AAAAAAA9AAAAAAAAADUAAAAAAAAANgAAAAAAAAA7AAAAAAAAADYAAAAAAAAANQAAAAAAAAA2AAAA"
    "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"
    "AIAAAAAAAAAArwAAAAAAAACCAAAAAAAAALgAAAAAAAAAgQAAAAAAAAC4AAAAAAAAAIIAAAAAAAAA"
    "nQAAAAAAAACCAAAAAAAAAIEAAAAAAAAArwAAAAAAAACdAAAAAAAAAIIAAAAAAAAAgwAAAAAAAACB"
    "AAAAAAAAALgAAAAAAAAAtwAAAAAAAACFAAAAAAAAAKMAAAAAAAAAogAAAAAAAACjAAAAAAAAAIUA"
    "AAAAAAAAhAAAAAAAAACDAAAAAAAAAIQAAAAAAAAAnQAAAAAAAACFAAAAAAAAAHIAAAAAAAAAnAAA"
    "AAAAAACGAAAAAAAAAJMAAAAAAAAAlAAAAAAAAACTAAAAAAAAAIYAAAAAAAAAcwAAAAAAAACIAAAA"
    "AAAAAJUAAAAAAAAAiQAAAAAAAACVAAAAAAAAAIgAAAAAAAAAlAAAAAAAAACJAAAAAAAAAJYAAAAA"
    "AAAAigAAAAAAAACWAAAAAAAAAIkAAAAAAAAAlQAAAAAAAACXAAAAAAAAAIoAAAAAAAAAlgAAAAAA"
    "AACKAAAAAAAAAJcAAAAAAAAAiwAAAAAAAACYAAAAAAAAAIwAAAAAAAAAlwAAAAAAAACMAAAAAAAA"
    "AJgAAAAAAAAAjQAAAAAAAABQAAAAAAAAAI4AAAAAAAAAjwAAAAAAAACMAAAAAAAAAIsAAAAAAAAA"
    "lwAAAAAAAACNAAAAAAAAAJgAAAAAAAAAmQAAAAAAAACOAAAAAAAAAJoAAAAAAAAAjwAAAAAAAACa"
    "AAAAAAAAAI4AAAAAAAAAmQAAAAAAAACOAAAAAAAAAI0AAAAAAAAAmQAAAAAAAACaAAAAAAAAAJAA"
    "AAAAAAAAjwAAAAAAAAClAAAAAAAAAG8AAAAAAAAApAAAAAAAAACeAAAAAAAAAJ8AAAAAAAAApgAA"
    "AAAAAACfAAAAAAAAAFcAAAAAAAAAoAAAAAAAAAByAAAAAAAAAKgAAAAAAAAApwAAAAAAAABdAAAA"
    "AAAAAKIAAAAAAAAAowAAAAAAAACiAAAAAAAAAKEAAAAAAAAAqQAAAAAAAACkAAAAAAAAAG8AAAAA"
    "AAAAugAAAAAAAACmAAAAAAAAAJ8AAAAAAAAApQAAAAAAAAByAAAAAAAAAKcAAAAAAAAAvAAAAAAA"
    "AACiAAAAAAAAAKkAAAAAAAAAqAAAAAAAAABAAAAAAAAAAKoAAAAAAAAAqwAAAAAAAAA3AAAAAAAA"
    "AKoAAAAAAAAAtAAAAAAAAABFAAAAAAAAAK0AAAAAAAAArAAAAAAAAACsAAAAAAAAADcAAAAAAAAA"
    "tQAAAAAAAABpAAAAAAAAALEAAAAAAAAAsAAAAAAAAACwAAAAAAAAALEAAAAAAAAArgAAAAAAAACz"
    "AAAAAAAAADYAAAAAAAAAtAAAAAAAAAA="
)


def _load_builtin_tris() -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(_BUILTIN_TRIS_B64)))


def load_msk(data: bytes) -> list[int]:
    text = data.decode("utf-8", errors="replace")
    entries = re.findall(r"\{(\d+)\s+(\d+)\}", text)
    if not entries:
        raise ValueError("No {index direction} entries found in .msk file.")
    point_indices = [int(idx) for idx, d in entries if d == "2"]
    if not point_indices:
        point_indices = [int(idx) for idx, _ in entries]
    return point_indices


def load_sym(data: bytes) -> np.ndarray:
    text = data.decode("utf-8", errors="replace").strip()
    return np.array([int(v) for v in text.split()], dtype=np.int32)


def symmetrize_landmarks(pts, sym):
    cx = pts[:, 0].mean()
    mirrored = pts.copy()
    mirrored[:, 0] = 2 * cx - mirrored[:, 0]
    return (pts + mirrored[sym]) / 2.0


def symmetrize_image(img, pts, sym, tris):
    h, w = img.shape[:2]
    sym_pts = symmetrize_landmarks(pts, sym)
    warped_orig = triangle_warp(img, pts, sym_pts, tris, h, w)
    flipped = img[:, ::-1].copy()
    flipped_pts = pts.copy()
    flipped_pts[:, 0] = w - 1 - flipped_pts[:, 0]
    flipped_pts = flipped_pts[sym]
    warped_flip = triangle_warp(flipped, flipped_pts, sym_pts, tris, h, w)
    return np.clip((warped_orig + warped_flip) / 2.0, 0, 255).astype(np.float32), sym_pts


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  YUV COLOR SPACE                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def rgb_to_yuv(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return np.stack([y, u, v], axis=-1)


def yuv_to_rgb(img, soft_clip=False):
    y, u, v = img[..., 0], img[..., 1], img[..., 2]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    rgb = np.stack([r, g, b], axis=-1)
    if soft_clip:
        return gamut_compress(rgb)
    return np.clip(rgb, 0, 255)


def gamut_compress(rgb):
    """Bring out-of-range RGB back into [0,255] while preserving hue.

    Per-channel clipping desaturates (when one channel pins at 255 and the
    others don't, the chroma ratio collapses toward gray). Instead, for any
    pixel that overshoots, scale the whole RGB vector toward its own luminance
    so the hue/chroma ratio is preserved and only saturation/lightness give.
    """
    rgb = rgb.astype(np.float32)
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
           + 0.114 * rgb[..., 2])[..., None]

    # Handle over-bright pixels: pull toward luminance until max channel <= 255
    hi = rgb.max(axis=-1, keepdims=True)
    over = hi > 255.0
    denom = np.where(hi - lum > 1e-6, hi - lum, 1.0)
    s_hi = np.where(over, (255.0 - lum) / denom, 1.0)
    s_hi = np.clip(s_hi, 0.0, 1.0)
    rgb = lum + (rgb - lum) * s_hi

    # Handle sub-zero pixels symmetrically
    lo = rgb.min(axis=-1, keepdims=True)
    under = lo < 0.0
    denom2 = np.where(lum - lo > 1e-6, lum - lo, 1.0)
    s_lo = np.where(under, lum / denom2, 1.0)
    s_lo = np.clip(s_lo, 0.0, 1.0)
    rgb = lum + (rgb - lum) * s_lo

    return np.clip(rgb, 0, 255)


def rgb_to_lab(img):
    """sRGB [0,255] -> CIE Lab (L in [0,100], a,b roughly [-127,127]).
    Uses OpenCV which handles sRGB gamma -> XYZ -> Lab correctly.
    Recommended over YUV for cross-race / cross-luminance transforms:
    L* is perceptually uniform so big luminance shifts don't wash out chroma."""
    img01 = (np.clip(img, 0, 255).astype(np.float32) / 255.0)
    return cv2.cvtColor(img01, cv2.COLOR_RGB2LAB)


def lab_to_rgb(img_lab):
    rgb = cv2.cvtColor(img_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb * 255.0, 0, 255)


# Color-space dispatch used by morph_controlled
_COLOR_SPACE_CONVERTERS = {
    "lab": (rgb_to_lab, lab_to_rgb),
    "yuv": (rgb_to_yuv, yuv_to_rgb),
    "rgb": (None, None),  # no conversion
}


def boost_skin_saturation(result_lab, mask, factor):
    """Multiply a*/b* by `factor` within the face mask, scaling about neutral
    (0,0): raises perceived colourfulness while preserving hue angle and
    lightness. factor=1.0 is a no-op; ~1.3 brings a CFD-average skin tone up to
    the saturation of a real individual face. CFD averages are desaturated
    because blending many faces cancels each person's colour variation, so any
    morph built from an average inherits that grayness. Use when the output
    looks ashen even though its lightness is correct."""
    if factor == 1.0:
        return result_lab
    blend = mask[..., None]
    result_lab = result_lab.copy()
    result_lab[..., 1:3] = (result_lab[..., 1:3] * (1.0 - blend)
                            + result_lab[..., 1:3] * factor * blend)
    return result_lab


# Landmarks the OpenFace-style crop fits on: jaw, pupils, brows.
_OF_JAW = [109, 110, 111, 125, 126, 127, 128, 129, 130, 131, 132, 133,
           114, 113, 112]
_OF_BROWS = list(range(71, 77)) + list(range(78, 83))
_OF_FIT = _OF_JAW + [0, 1] + _OF_BROWS
# _OF_FIT positions in a 512 px frame: mean shape of the 364 input24 faces,
# eye-normalized, eyes at y = 0.255*512, lowest jaw point at y = 490.
_OF_TEMPLATE = np.array([
    (44.93, 127.96),
    (42.30, 202.09),
    (52.29, 278.41),
    (63.84, 333.76),
    (83.22, 377.34),
    (133.45, 437.05),
    (190.25, 475.26),
    (253.62, 490.00),
    (318.03, 475.05),
    (374.41, 439.89),
    (422.87, 385.81),
    (444.76, 340.35),
    (458.14, 282.44),
    (468.61, 204.62),
    (466.27, 129.98),
    (159.48, 130.56),
    (352.52, 130.56),
    (73.32, 106.86),
    (94.35, 79.41),
    (125.49, 65.55),
    (166.13, 69.58),
    (210.71, 80.45),
    (216.69, 104.19),
    (302.49, 78.22),
    (349.34, 69.16),
    (386.99, 65.17),
    (419.33, 79.08),
    (438.11, 107.88)
], dtype=np.float64)


def openface_crop(img, pts, out_size=512, mask=True, chin_max=500,
                  brow_up=0.8, **_legacy):
    """Square OpenFace-style crop, masked to the face over black.

    Least-squares similarity fit of jaw + pupils + brows onto a fixed
    template, so the whole face outline (not just the eyes) is standardized.
    If the chin would fall below chin_max (of 512), the fit is shrunk about
    the eye midpoint. With mask=True everything outside the hull of the jaw
    and the brows raised by brow_up*IPD is set to 0.
    Returns (cropped_img, remapped_pts). Old zoom/eye_y_frac args are
    accepted and ignored.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= max(_OF_FIT):
        raise ValueError("OpenFace crop needs the 189-point CFD template.")
    M = _umeyama_similarity(pts[_OF_FIT], _OF_TEMPLATE)
    q = pts @ M[:, :2].T + M[:, 2]
    low = q[_OF_JAW][:, 1].max()
    if low > chin_max:
        c = (q[0] + q[1]) / 2
        r = (chin_max - c[1]) / (low - c[1])
        M = np.hstack([r * M[:, :2], (r * M[:, 2] + (1 - r) * c)[:, None]])
    sc = out_size / 512.0
    M = M * sc
    out = cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_CUBIC,
                         borderValue=0)
    new_pts = pts @ M[:, :2].T + M[:, 2]
    if mask:
        ipd = np.linalg.norm(new_pts[1] - new_pts[0])
        brows = new_pts[_OF_BROWS].copy()
        brows[:, 1] -= brow_up * ipd
        hull = cv2.convexHull(
            np.vstack([new_pts[_OF_JAW], brows]).astype(np.int32))
        m = np.zeros((out_size, out_size), np.uint8)
        cv2.fillConvexPoly(m, hull, 255)
        k = max(1, int(round(5 * sc)))
        m = cv2.erode(m, np.ones((k, k), np.uint8))
        out[m == 0] = 0
    return out, new_pts.astype(np.float32)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  CORE:  WARP / MASK / NORMALIZE                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def _umeyama_similarity(src, dst):
    """Closed-form least-squares similarity (rotation + uniform scale + shift).

    Umeyama (1991), reflection suppressed. Alternative to
    cv2.estimateAffinePartial2D(..., LMEDS): with only three correspondences
    LMEDS is a robust estimator with nothing to be robust about — it fits a
    minimal two-point subset exactly and leaves the third anchor with a large
    residual, and its inlier choice is not translation-invariant. This uses
    all three anchors optimally and is exact and deterministic.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Xd.T @ Xs / len(src))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1.0
    R = U @ D @ Vt
    var_s = (Xs ** 2).sum() / len(src)
    c = float((S * np.diag(D)).sum() / var_s) if var_s > 0 else 1.0
    out = np.empty((2, 3), np.float64)
    out[:, :2] = c * R
    out[:, 2] = mu_d - (c * R) @ mu_s
    return out.astype(np.float32)


def normalize_to_reference(img, pts, ref_pts, norm_indices=(0, 1, 96),
                           align="umeyama"):
    """Similarity-align `pts` (and `img`) onto `ref_pts` at the anchor points.

    `align="umeyama"` (default) is the closed-form least-squares fit; see
    _umeyama_similarity. `align="lmeds"` reproduces stimuli generated before
    the 2026-10-04 cutover.
    """
    src = np.float32([pts[i] for i in norm_indices])
    dst = np.float32([ref_pts[i] for i in norm_indices])
    if align == "umeyama":
        M = _umeyama_similarity(src, dst)
    else:
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return img.copy(), pts.copy()
    h, w = img.shape[:2]
    img_out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_out = (M @ np.hstack([pts, ones]).T).T.astype(np.float32)
    return img_out, pts_out


def _warp_triangle(img, src_pts, dst_pts, tri_indices):
    if len(set(tri_indices)) < 3:
        return None, None, None
    src_tri = np.float32([src_pts[i] for i in tri_indices])
    dst_tri = np.float32([dst_pts[i] for i in tri_indices])
    r1 = cv2.boundingRect(src_tri[None])
    r2 = cv2.boundingRect(dst_tri[None])
    if r1[2] <= 1 or r1[3] <= 1 or r2[2] <= 1 or r2[3] <= 1:
        return None, None, None
    y1, y1e = max(r1[1], 0), max(r1[1] + r1[3], 0)
    x1 = max(r1[0], 0)
    src_crop = img[y1:y1e, x1:x1 + r1[2]]
    if src_crop.size == 0:
        return None, None, None
    src_off = np.ascontiguousarray(
        src_tri - np.float32([r1[0], r1[1]]), dtype=np.float32)
    dst_off = np.ascontiguousarray(
        dst_tri - np.float32([r2[0], r2[1]]), dtype=np.float32)
    if src_off.shape != (3, 2) or dst_off.shape != (3, 2):
        return None, None, None
    M = cv2.getAffineTransform(src_off, dst_off)
    warped = cv2.warpAffine(src_crop, M, (r2[2], r2[3]),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_off.astype(np.int32), 255, lineType=cv2.LINE_8)
    # Dilate by 1px so adjacent triangles overlap on their shared edge.
    # Overlap is harmless because triangle_warp normalizes by accumulated mask.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    mask_f = np.repeat(mask_f, 3, axis=2)
    return warped.astype(np.float32) * mask_f, mask_f, r2


def triangle_warp(img, src_pts, dst_pts, tris, h, w):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    out = np.zeros((h, w, 3), dtype=np.float32)
    acc = np.zeros((h, w, 3), dtype=np.float32)
    for tri in tris:
        pts_dst = [dst_pts[i] for i in tri]
        # Gate on DESTINATION area, not source. Foreshortened source triangles
        # (nose sides, eye corners, oblique jaw) can be tiny in the source yet
        # cover real area in the output; skipping them on source area left
        # uncovered (black) regions. Only skip genuinely degenerate triangles.
        area = 0.5 * abs(
            pts_dst[0][0] * (pts_dst[1][1] - pts_dst[2][1])
            + pts_dst[1][0] * (pts_dst[2][1] - pts_dst[0][1])
            + pts_dst[2][0] * (pts_dst[0][1] - pts_dst[1][1]))
        if area < 1e-6:
            continue
        result = _warp_triangle(img, src_pts, dst_pts, tri)
        if result[0] is None:
            continue
        warped_tri, mask_f, r = result
        x, y, w_, h_ = r
        sy = sx = 0
        dy, dx = y, x
        if y < 0: sy = -y; dy = 0
        if x < 0: sx = -x; dx = 0
        hc = min(h_, warped_tri.shape[0] - sy, h - dy)
        wc = min(w_, warped_tri.shape[1] - sx, w - dx)
        if hc <= 0 or wc <= 0:
            continue
        out[dy:dy + hc, dx:dx + wc] += warped_tri[sy:sy + hc, sx:sx + wc]
        acc[dy:dy + hc, dx:dx + wc] += mask_f[sy:sy + hc, sx:sx + wc]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(acc > 1e-3, out / (acc + 1e-7), 0.0)
    return np.clip(out, 0, 255)


def make_face_mask(h, w, pts, indices, feather=25):
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = pts[indices].astype(np.int32)
    # fillPoly (not fillConvexPoly): a jaw/face outline is CONCAVE, so the
    # convex version cut straight chords across the chin and jaw, which the
    # feather then spread unevenly. fillPoly respects the true outline.
    cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_AA)
    if feather > 0:
        # Erode first so the feathered falloff sits just INSIDE the true face
        # edge rather than straddling it (prevents the bright/dark jaw ring).
        er = max(1, feather // 3)
        mask = cv2.erode(mask, np.ones((er, er), np.uint8), iterations=1)
        k = (feather * 2) | 1
        mask = cv2.GaussianBlur(mask, (k, k), feather / 2)
    return mask.astype(np.float32) / 255.0


def blur_lip_seam(img, pts, seam_indices, thickness=8, blur_ksize=15):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    seam = np.array([pts[i] for i in seam_indices],
                    dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [seam], isClosed=False, color=255, thickness=thickness)
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    return (img * (1 - alpha) + blurred * alpha).astype(np.uint8)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  TRANSFORM  (C + strength * (B − A))                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def _edge_inner_mask(h, w, pts, indices, width):
    """Face polygon eroded by `width` px (1 inside, 0 outside)."""
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts[indices].astype(np.int32)], 1)
    k = 2 * int(round(width)) + 1
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(np.float32)


def _fill_from_interior(x, inner, sigma):
    """Replace values outside `inner` with a normalized-convolution average of
    nearby interior values, so pixels at the face outline (which mix in hair or
    background) do not set the color difference there."""
    wgt = cv2.GaussianBlur(inner, (0, 0), sigma)
    num = cv2.GaussianBlur(x * inner[..., None], (0, 0), sigma)
    filled = num / np.maximum(wgt, 1e-6)[..., None]
    return np.where(inner[..., None] > 0.5, x, filled)


def morph_controlled(
    imgA, ptsA, imgB, ptsB, imgC, ptsC, tris,
    shape_strength=0.7, color_strength=0.0, texture_strength=0.5,
    face_poly_indices=None, feather=25, norm_indices=(0, 1, 96),
    texture_blur_sigma=30, mask_output=True, use_yuv=False,
    chroma_boost=1.0, match_exposure=True, bg_color=(128, 128, 128),
    color_anchor="subject", texture_anchor="subject",
    color_space=None, texture_luma_only=True, align="umeyama", edge_fill=0.0,
):
    # Color space. Default to plain RGB: the additive cross-category delta
    # warped_C + color*(warped_B - warped_A) keeps the per-channel ratios, so
    # skin stays warm. Doing the same delta in Lab/YUV SUBTRACTS chroma in
    # shadow/dark regions (a dark exemplar has lower absolute chroma than its
    # average, so the residual is negative) -> the ashen/gray look. `use_yuv`
    # is kept for backward-compat; `color_space` overrides it when given.
    if color_space is None:
        color_space = "yuv" if use_yuv else "rgb"
    to_cs, from_cs = _COLOR_SPACE_CONVERTERS[color_space]
    use_cs = to_cs is not None
    # `chroma_boost` is reused as the Lab skin-saturation factor: it scales
    # a*/b* about neutral inside the face so the result isn't ashen.
    skin_saturation = chroma_boost

    h, w = imgC.shape[:2]
    imgB_n, ptsB_n = normalize_to_reference(imgB, ptsB, ptsA, norm_indices,
                                            align=align)
    pts_target = ptsC + shape_strength * (ptsB_n - ptsA)
    warped_C = triangle_warp(imgC, ptsC, pts_target, tris, h, w)

    # Edge fill (v1.2.0): the outermost face pixels of the source photograph and
    # the averages mix in hair or background, which put a light rim on the
    # transferred color and texture. With edge_fill > 0, color differences in a
    # band of that width (fraction of face width) are filled in from interior
    # skin, texture is computed from interior pixels and faded out before the
    # outline, and the finished face is filled in across the band. edge_fill = 0
    # reproduces v1.1.0 output exactly.
    inner = soft_inner = None
    band = 0.0
    if edge_fill and face_poly_indices is not None:
        band = max(2.0, edge_fill * float(np.ptp(pts_target[face_poly_indices][:, 0])))
        inner = _edge_inner_mask(h, w, pts_target, face_poly_indices, band)
        soft_inner = cv2.GaussianBlur(inner, (0, 0), band / 3)

    def _fill(x):
        # A narrow kernel copies the skin just inside the band, preserving the
        # natural darkening toward the outline.
        return x if inner is None else _fill_from_interior(x, inner, band / 3)

    apply_color = (color_strength != 0.0) or (color_anchor in ("target", "residual"))
    apply_texture = (texture_strength != 0.0) or (texture_anchor in ("target", "residual"))
    needs_B = apply_texture or apply_color
    if needs_B:
        warped_B = triangle_warp(imgB_n, ptsB_n, pts_target, tris, h, w)

    # Convert to working color space
    if use_cs and needs_B:
        warped_C_cs = to_cs(warped_C)
        warped_B_cs = to_cs(warped_B)
    elif needs_B:
        warped_C_cs, warped_B_cs = warped_C, warped_B

    # Color delta — a simple additive transfer in a perceptually uniform space.
    # At color_strength=1 the subject and residual anchors are identical and
    # equal  Subject + (target_avg - from_avg) = target_avg + (Subject - from_avg),
    # i.e. the identity-preserving cross-race transform.
    warped_A = None
    warped_A_cs = None
    if apply_color:
        warped_A = triangle_warp(imgA, ptsA, pts_target, tris, h, w)
        warped_A_cs = to_cs(warped_A) if use_cs else warped_A

        if color_anchor == "residual":
            # Plant the subject's deviation from the FROM-average onto the
            # TARGET average:  result = target_avg + color * (Subject - from_avg)
            #   strength 0 -> target-average color, strength 1 -> identity transfer
            result_cs = warped_B_cs + color_strength * _fill(warped_C_cs - warped_A_cs)
        elif color_anchor == "target":
            # Anchor at TARGET coloring at strength 0, blend back to the
            # subject's own coloring as strength -> 1.
            white_colored = warped_C_cs + _fill(warped_B_cs - warped_A_cs)
            result_cs = (white_colored
                         + color_strength * (warped_C_cs - white_colored))
        else:
            # Default: anchor at the subject, move toward target as strength->1.
            result_cs = warped_C_cs + color_strength * _fill(warped_B_cs - warped_A_cs)
    else:
        result_cs = warped_C_cs.copy() if needs_B else warped_C.copy()

    # High-freq texture blend
    if apply_texture and face_poly_indices is not None:
        face_mask = make_face_mask(h, w, pts_target, face_poly_indices,
                                   feather=feather)
        ksize = int(texture_blur_sigma * 6) | 1

        def _high(x):
            # High-frequency detail. With texture_luma_only the detail is reduced
            # to luminance and applied equally to all channels, so it adds pore /
            # surface relief WITHOUT carrying the (low-chroma) micro-colour of the
            # source skin — that micro-colour is what speckles a lightened face
            # back toward gray/ashen.
            if inner is None:
                hi = x - cv2.GaussianBlur(x, (ksize, ksize), texture_blur_sigma)
            else:
                # Blur only interior pixels so background cannot leak into the
                # detail, then fade the detail out before the outline.
                wgt = cv2.GaussianBlur(inner, (ksize, ksize), texture_blur_sigma)
                low = cv2.GaussianBlur(x * inner[..., None], (ksize, ksize),
                                       texture_blur_sigma) / np.maximum(wgt, 1e-6)[..., None]
                hi = (x - low) * soft_inner[..., None]
            if texture_luma_only:
                if color_space == "rgb":
                    lum = (0.299 * hi[..., 0] + 0.587 * hi[..., 1]
                           + 0.114 * hi[..., 2])
                    hi = np.repeat(lum[..., None], 3, axis=2)
                else:               # Lab/YUV: channel 0 is luma; drop chroma detail
                    hi = hi.copy(); hi[..., 1] = 0.0; hi[..., 2] = 0.0
            return hi

        highB = _high(warped_B_cs)
        if texture_anchor == "residual":
            # Identity-transfer for texture: keep the subject's own high-freq
            # detail rather than the target average's smoothness.
            if warped_A_cs is None:
                warped_A = triangle_warp(imgA, ptsA, pts_target, tris, h, w)
                warped_A_cs = to_cs(warped_A) if use_cs else warped_A
            highC = _high(warped_C_cs)
            highA = _high(warped_A_cs)
            t = texture_strength * face_mask[..., None]
            result_cs = result_cs + t * (highC - highA)
        elif texture_anchor == "target":
            # strength 0 -> full target texture, strength 1 -> none added
            t = (1.0 - texture_strength) * face_mask[..., None]
            result_cs = result_cs + t * highB
        else:
            # strength 0 -> none added, strength 1 -> full target texture
            t = texture_strength * face_mask[..., None]
            result_cs = result_cs + t * highB

    # With edge fill, the final face is also filled in from the interior across
    # the edge band, so outline pixels of the averages (which blend in their
    # hair and background) cannot tint the rim either.
    if inner is not None and needs_B:
        result_cs = _fill(result_cs)

    # Skin saturation boost (Lab only): CFD averages are desaturated (blending
    # many faces cancels each person's color), so any morph built from an
    # average inherits that grayness. Lifting a*/b* in the face restores a
    # believable skin tone without touching lightness or hue.
    if (skin_saturation != 1.0 and color_space == "lab" and needs_B
            and face_poly_indices is not None):
        smask = make_face_mask(h, w, pts_target, face_poly_indices,
                               feather=feather)
        result_cs = boost_skin_saturation(result_cs, smask, skin_saturation)

    # Convert back. For YUV use the gamut-preserving soft clip so out-of-range
    # pixels desaturate toward luminance instead of per-channel clipping.
    if not needs_B:
        result = result_cs
    elif color_space == "yuv":
        result = yuv_to_rgb(result_cs, soft_clip=True)
    else:
        result = from_cs(result_cs) if use_cs else result_cs
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Outer mask — composite over a background instead of multiplying into
    # black. Multiplying by a feathered (0..1) mask darkens the feather zone
    # toward black, producing the dark/uneven ring around the jaw and a muddy
    # overall look. Alpha-compositing over a neutral bg keeps the falloff clean.
    if mask_output and face_poly_indices is not None:
        outer = make_face_mask(h, w, pts_target, face_poly_indices,
                               feather=feather)[..., None]
        bg = np.full_like(result, bg_color, dtype=np.float32)
        result = np.clip(result.astype(np.float32) * outer + bg * (1.0 - outer),
                         0, 255).astype(np.uint8)

    return result, pts_target


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  MORPH SEQUENCE  (frame-by-frame A ↔ B)                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def morph_frame(imgA, ptsA, imgB, ptsB, tris, alpha, h, w):
    """Single morph frame at position `alpha` (0 = A, 1 = B)."""
    pts_morph = (1.0 - alpha) * ptsA + alpha * ptsB
    warped_A = triangle_warp(imgA, ptsA, pts_morph, tris, h, w)
    warped_B = triangle_warp(imgB, ptsB, pts_morph, tris, h, w)
    blended = (1.0 - alpha) * warped_A + alpha * warped_B
    return np.clip(blended, 0, 255), pts_morph


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  FACE AVERAGING                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def average_faces(images, all_pts, tris, norm_indices=(0, 1, 96),
                  face_poly=None, feather=25, pad=100,
                  mask_output=True, bg_color=(128, 128, 128), align="umeyama"):
    """Compute the average face from a list of images and their landmarks.

    1. Pad all images
    2. Normalize all faces to the first face's coordinate frame
    3. Compute mean shape
    4. Warp every face to the mean shape
    5. Average pixel values
    6. Optionally mask to face polygon

    Returns (avg_image, avg_pts) after cropping padding back out.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images to average.")
    if n == 1:
        return images[0].astype(np.uint8), all_pts[0]

    h0, w0 = images[0].shape[:2]

    # Resize all to match first image
    imgs, pts_list = [], []
    for i in range(n):
        img, pts = images[i], all_pts[i].copy()
        hi, wi = img.shape[:2]
        if hi != h0 or wi != w0:
            sx, sy = w0 / wi, h0 / hi
            img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_LINEAR)
            pts = pts * np.array([sx, sy], dtype=np.float32)
        imgs.append(img)
        pts_list.append(pts)

    # Pad
    padded_imgs = []
    padded_pts = []
    for img, pts in zip(imgs, pts_list):
        img_p = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                   cv2.BORDER_REFLECT_101)
        padded_imgs.append(img_p)
        padded_pts.append(pts + pad)

    # Normalize all to the first face's frame
    ref_pts = padded_pts[0]
    norm_imgs = [padded_imgs[0]]
    norm_pts = [padded_pts[0]]
    for i in range(1, n):
        img_n, pts_n = normalize_to_reference(padded_imgs[i], padded_pts[i],
                                              ref_pts, norm_indices,
                                              align=align)
        norm_imgs.append(img_n)
        norm_pts.append(pts_n)

    # Mean shape
    avg_pts = np.mean(norm_pts, axis=0)

    # Warp each face to mean shape and accumulate
    h, w = norm_imgs[0].shape[:2]
    accumulator = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(n):
        warped = triangle_warp(norm_imgs[i], norm_pts[i], avg_pts, tris, h, w)
        accumulator += warped.astype(np.float64)
    avg_img = (accumulator / n)

    avg_img = np.clip(avg_img, 0, 255).astype(np.uint8)

    # Mask — composite over neutral bg (see note in morph_controlled).
    if mask_output and face_poly is not None:
        outer = make_face_mask(h, w, avg_pts, face_poly, feather=feather)[..., None]
        bg = np.full_like(avg_img, bg_color, dtype=np.float32)
        avg_img = np.clip(avg_img.astype(np.float32) * outer + bg * (1.0 - outer),
                          0, 255).astype(np.uint8)

    # Lip seam
    avg_img = blur_lip_seam(avg_img, avg_pts, LIP_SEAM_INDICES,
                            thickness=10, blur_ksize=23)

    # Crop padding
    avg_img = avg_img[pad:-pad, pad:-pad]
    avg_pts = avg_pts - pad

    return avg_img, avg_pts


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  JITTER  (random shape-space perturbation)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def load_training_shapes(folder: str, expected_landmarks=189):
    """Load all .tem files in `folder` (recursively) as training shapes."""
    tem_files = sorted(glob.glob(os.path.join(folder, "**", "*.tem"),
                                 recursive=True))
    shapes = []
    for p in tem_files:
        try:
            pts, _ = load_tem(p, expected_landmarks)
            shapes.append(pts)
        except Exception:
            continue
    if not shapes:
        raise ValueError(f"No valid .tem files found in {folder}")
    return np.array(shapes)


def sample_jitter_direction(training_shapes, rng, percentile=50,
                            n_samples=200):
    """Sample a random direction in shape space and a reference scale."""
    n = len(training_shapes)
    if n < 2:
        raise ValueError("Need at least 2 training shapes for jitter")
    i1, i2 = rng.choice(n, size=2, replace=False)
    diff = training_shapes[i2].ravel() - training_shapes[i1].ravel()
    direction = diff / (np.linalg.norm(diff) + 1e-10)
    k = min(n_samples, n * (n - 1) // 2)
    dists = [
        np.linalg.norm(
            training_shapes[rng.choice(n)].ravel()
            - training_shapes[rng.choice(n)].ravel()
        )
        for _ in range(k)
    ]
    return direction, np.percentile(dists, percentile)


def center_face(img, pts, indices):
    """Translate so the face polygon centroid is at image center."""
    h, w = img.shape[:2]
    shift = np.array([w / 2, h / 2], dtype=np.float32) - pts[indices].mean(0)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img_out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return img_out, pts + shift


def apply_bg_mask(img_rgb, mask_f, bg_color=(128, 128, 128)):
    """Composite image onto solid background using float mask [0,1]."""
    bg = np.full_like(img_rgb, bg_color, dtype=np.float32)
    m = mask_f[..., None] if mask_f.ndim == 2 else mask_f
    out = img_rgb.astype(np.float32) * m + bg * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter_single(img, pts, training_shapes, tris, rng,
                  alpha_fraction=0.5, percentile=50, pad=20,
                  face_poly=None, feather=10, bg_color=(128, 128, 128),
                  center=True):
    """Apply a random shape-space jitter to a single face.

    Returns (jittered_img, jittered_pts, meta_dict).
    """
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                             cv2.BORDER_REFLECT_101)
    pts = pts + pad
    N = pts.shape[0]

    direction, ref_scale = sample_jitter_direction(
        training_shapes, rng, percentile)
    alpha = alpha_fraction * ref_scale
    sign = 1 if rng.random() < 0.5 else -1
    pts_jittered = (pts.ravel() + sign * alpha * direction).reshape(N, 2)

    h, w = img.shape[:2]
    out = triangle_warp(img, pts, pts_jittered, tris, h, w)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Crop padding
    out = out[pad:-pad, pad:-pad]
    pts_jittered = pts_jittered - pad

    # Center and mask
    if face_poly is not None:
        if center:
            out, pts_jittered = center_face(out, pts_jittered, face_poly)
        mask = make_face_mask(out.shape[0], out.shape[1], pts_jittered,
                              face_poly, feather=feather)
        out = apply_bg_mask(out, mask, bg_color)

    meta = {"alpha_px": round(float(alpha), 2),
            "ref_scale": round(float(ref_scale), 2),
            "sign": "+" if sign > 0 else "-"}
    return out, pts_jittered, meta


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  REVERSE CORRELATION STIMULI  (PCA ± perturbation pairs))                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


class SimplePCA:
    """Lightweight PCA using only NumPy (no sklearn dependency)."""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        if n_samples < n_features:
            cov = (Xc @ Xc.T) / (n_samples - 1)
            evals, evecs = np.linalg.eigh(cov)
            evecs = Xc.T @ evecs
            for i in range(evecs.shape[1]):
                evecs[:, i] /= np.linalg.norm(evecs[:, i])
        else:
            cov = (Xc.T @ Xc) / (n_samples - 1)
            evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        nc = min(self.n_components, len(evals), n_features)
        self.n_components_ = nc
        self.components_ = evecs[:, :nc].T
        self.explained_variance_ = evals[:nc]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(evals)
        return self


def procrustes_align(shape, reference):
    """Align shape to reference via Procrustes (rotation, scale, translation)."""
    sc = shape - shape.mean(0)
    rc = reference - reference.mean(0)
    sn, rn = np.linalg.norm(sc), np.linalg.norm(rc)
    if sn < 1e-10 or rn < 1e-10:
        return shape
    sc, rc = sc / sn, rc / rn
    U, _, Vt = np.linalg.svd(sc.T @ rc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return (sc @ R) * rn + reference.mean(0)


def build_shape_pca(folder: str, expected_landmarks=189, n_components=50):
    """Build a PCA model from all .tem files in a folder."""
    tem_files = sorted(glob.glob(os.path.join(folder, "**", "*.tem"),
                                  recursive=True))
    shapes = []
    for p in tem_files:
        try:
            pts, _ = load_tem(p, expected_landmarks)
            shapes.append(pts)
        except Exception:
            continue
    if len(shapes) < 3:
        raise ValueError(f"Need at least 3 shapes for PCA, found {len(shapes)}")

    # Procrustes alignment (3 iterations)
    ref = shapes[0].copy()
    for _ in range(3):
        aligned = [procrustes_align(s, ref) for s in shapes]
        ref = np.mean(aligned, axis=0)
    X = np.array(aligned).reshape(len(aligned), -1)

    nc = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = SimplePCA(n_components=nc)
    pca.fit(X)
    return pca, ref, len(shapes)


def sample_pca_direction(pca, n_pts, rng, n_active_pcs=5, target_px=12.0):
    """Sample a random direction in PC space scaled to target_px displacement."""
    nap = min(n_active_pcs, pca.n_components_)
    active = rng.choice(pca.n_components_, size=nap, replace=False)
    weights = rng.standard_normal(nap)
    weights /= np.linalg.norm(weights)
    direction = np.zeros(pca.components_.shape[1])
    for idx, w in zip(active, weights):
        direction += w * pca.components_[idx]
    # Scale so mean per-landmark displacement = target_px
    vecs = direction.reshape(n_pts, 2)
    disp = np.mean(np.linalg.norm(vecs, axis=1))
    alpha = target_px / (disp + 1e-8)
    return direction, alpha, active.tolist(), weights.tolist()


def generate_pair(img, pts, direction, alpha, tris, pad,
                  face_poly=None, feather=10, bg_color=(0, 0, 0),
                  grayscale=False):
    """Generate a +/- pair from a baseline image along a shape direction."""
    # BORDER_REPLICATE (not REFLECT_101): reflection mirrors edge pixels back
    # into the face at large perturbations, and because many warped variants
    # get averaged in RC analysis, that systematic edge artifact can leak into
    # the recovered classification image. Replication does not inject mirrored
    # content.
    img_p = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    pts_p = pts + pad
    N = pts_p.shape[0]
    h, w = img_p.shape[:2]
    s0 = pts_p.ravel()

    pts_plus = (s0 + alpha * direction).reshape(N, 2)
    pts_minus = (s0 - alpha * direction).reshape(N, 2)

    warped_plus = triangle_warp(img_p, pts_p, pts_plus, tris, h, w)
    warped_minus = triangle_warp(img_p, pts_p, pts_minus, tris, h, w)

    warped_plus = np.clip(warped_plus, 0, 255).astype(np.uint8)
    warped_minus = np.clip(warped_minus, 0, 255).astype(np.uint8)

    # Crop padding
    warped_plus = warped_plus[pad:-pad, pad:-pad]
    warped_minus = warped_minus[pad:-pad, pad:-pad]
    pts_plus = pts_plus - pad
    pts_minus = pts_minus - pad

    # Mask and center
    if face_poly is not None:
        warped_plus, pts_plus = center_face(warped_plus, pts_plus, face_poly)
        warped_minus, pts_minus = center_face(warped_minus, pts_minus,
                                              face_poly)
        mask_p = make_face_mask(warped_plus.shape[0], warped_plus.shape[1],
                                pts_plus, face_poly, feather=feather)
        mask_m = make_face_mask(warped_minus.shape[0], warped_minus.shape[1],
                                pts_minus, face_poly, feather=feather)
        warped_plus = apply_bg_mask(warped_plus, mask_p, bg_color)
        warped_minus = apply_bg_mask(warped_minus, mask_m, bg_color)

    if grayscale:
        warped_plus = cv2.cvtColor(warped_plus, cv2.COLOR_RGB2GRAY)
        warped_minus = cv2.cvtColor(warped_minus, cv2.COLOR_RGB2GRAY)

    return warped_plus, warped_minus


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  CLOUD HELPERS                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def _show_preview(uploaded_file, width=120):
    """Show a small thumbnail of an uploaded image file."""
    if uploaded_file is not None:
        data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        uploaded_file.seek(0)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=width)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def read_image(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    uploaded.seek(0)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {uploaded.name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def encode_jpg(img_rgb, quality=95):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def save_tem_to_bytes(pts, curve_block=None):
    buf = io.StringIO()
    buf.write(f"{len(pts)}\n")
    for x, y in pts:
        buf.write(f"{x:.4f}\t{y:.4f}\n")
    if curve_block:
        buf.writelines(curve_block)
    return buf.getvalue().encode("utf-8")


def pair_uploaded_files(files, expected_landmarks=189):
    """Group uploaded files by stem → list of (stem, img_data, tem_data).
    img_data and tem_data are the raw UploadedFile objects."""
    images, tems = {}, {}
    for f in files:
        stem = os.path.splitext(f.name)[0]
        if f.name.lower().endswith(".tem"):
            tems[stem] = f
        else:
            images[stem] = f
    paired = []
    for stem in sorted(images):
        if stem in tems:
            paired.append((stem, images[stem], tems[stem]))
    return paired


def single_face_upload(files, expected_landmarks=189):
    """Split one combined upload into (image_file, tem_file) for ONE face.

    Lets A and C use the same "drop the image and its .tem together" box as B,
    instead of two separate boxes per face. Pairs by filename stem like
    pair_uploaded_files, but falls back to pairing a lone image with a lone
    .tem whatever they are named — an average and its template are often
    exported under different stems, which the old split boxes tolerated.
    Either half may come back None so the caller's "upload X" messages still
    say which piece is missing.
    """
    files = files or []
    imgs = [f for f in files if not f.name.lower().endswith(".tem")]
    tems = [f for f in files if f.name.lower().endswith(".tem")]
    if len(imgs) == 1 and len(tems) == 1:
        return imgs[0], tems[0]
    paired = pair_uploaded_files(files, expected_landmarks)
    if paired:
        return paired[0][1], paired[0][2]
    return (imgs[0] if imgs else None), (tems[0] if tems else None)


def pair_doppelganger_files(files, suffix_a, suffix_b):
    """Group uploaded files into identity pairs by suffix."""
    images, tems = {}, {}
    for f in files:
        base = os.path.splitext(f.name)[0]
        if f.name.lower().endswith(".tem"):
            tems[base] = f
        else:
            images[base] = f
    identities = {}
    for base in images:
        if base.endswith(suffix_a):
            identity = base[:-len(suffix_a)]
            identities.setdefault(identity, {})["a_img"] = images[base]
            if base in tems:
                identities[identity]["a_tem"] = tems[base]
        elif base.endswith(suffix_b):
            identity = base[:-len(suffix_b)]
            identities.setdefault(identity, {})["b_img"] = images[base]
            if base in tems:
                identities[identity]["b_tem"] = tems[base]
    complete = []
    for identity in sorted(identities):
        d = identities[identity]
        if all(k in d for k in ("a_img", "a_tem", "b_img", "b_tem")):
            complete.append((identity, d["a_img"], d["a_tem"],
                             d["b_img"], d["b_tem"]))
    return complete


def load_training_shapes_from_uploads(tem_files, expected_landmarks=189):
    """Load .tem uploads into a training shapes array."""
    shapes = []
    for f in tem_files:
        try:
            pts, _ = load_tem_from_bytes(f.read(), expected_landmarks)
            f.seek(0)
            shapes.append(pts)
        except Exception:
            f.seek(0)
            continue
    if not shapes:
        raise ValueError("No valid .tem files found in uploads.")
    return np.array(shapes)


def build_shape_pca_from_uploads(tem_files, expected_landmarks=189,
                                 n_components=50):
    """Build PCA from uploaded .tem files."""
    shapes = []
    for f in tem_files:
        try:
            pts, _ = load_tem_from_bytes(f.read(), expected_landmarks)
            f.seek(0)
            shapes.append(pts)
        except Exception:
            f.seek(0)
            continue
    if len(shapes) < 3:
        raise ValueError(f"Need at least 3 shapes for PCA, found {len(shapes)}")
    ref = shapes[0].copy()
    for _ in range(3):
        aligned = [procrustes_align(s, ref) for s in shapes]
        ref = np.mean(aligned, axis=0)
    X = np.array(aligned).reshape(len(aligned), -1)
    nc = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = SimplePCA(n_components=nc)
    pca.fit(X)
    return pca, ref, len(shapes)


def make_zip(file_dict):
    """Create a zip from {filename: bytes} dict, return bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in file_dict.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  SINGLE-IMAGE RUNNERS                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝


def run_transform_single(imgA, ptsA, imgB, ptsB, imgC, ptsC,
                         curve_block, tris, params):
    pad = params["pad"]
    norm_indices = params["norm_indices"]
    align = params.get("align", "lmeds")
    hA, wA = imgA.shape[:2]

    hB, wB = imgB.shape[:2]
    if hB != hA or wB != wA:
        sx, sy = wA / wB, hA / hB
        imgB = cv2.resize(imgB, (wA, hA), interpolation=cv2.INTER_LINEAR)
        ptsB = ptsB * np.array([sx, sy], dtype=np.float32)

    _imgA = cv2.copyMakeBorder(imgA, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    _imgC = cv2.copyMakeBorder(imgC, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    _imgB = cv2.copyMakeBorder(imgB, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    _ptsA, _ptsC, _ptsB = ptsA + pad, ptsC + pad, ptsB + pad

    _imgC, _ptsC = normalize_to_reference(_imgC, _ptsC, _ptsA, norm_indices,
                                          align=align)

    morphed, target_pts = morph_controlled(
        _imgA, _ptsA, _imgB, _ptsB, _imgC, _ptsC, tris,
        shape_strength=params["shape"],
        color_strength=params["color"],
        texture_strength=params["texture"],
        face_poly_indices=params["face_poly"],
        feather=params["feather"],
        norm_indices=norm_indices,
        align=align,
        texture_blur_sigma=params["blur_sigma"],
        mask_output=params["mask_output"],
        use_yuv=params["use_yuv"],
        bg_color=params.get("bg_color", (128, 128, 128)),
        color_anchor=params.get("color_anchor", "subject"),
        texture_anchor=params.get("texture_anchor", "subject"),
        chroma_boost=params.get("chroma_boost", 1.0),
        color_space=params.get("color_space"),
        edge_fill=params.get("edge_fill", 0.0),
    )
    morphed = morphed[pad:-pad, pad:-pad]
    target_pts = target_pts - pad
    morphed = blur_lip_seam(morphed, target_pts, LIP_SEAM_INDICES,
                            thickness=10, blur_ksize=23)
    if params.get("of_crop"):
        morphed, target_pts = openface_crop(
            morphed, target_pts,
            out_size=params.get("of_size", 512),
            mask=params.get("of_mask", True))
    return morphed, target_pts


def run_morph_sequence_single(imgA, ptsA, imgB, ptsB, tris, params):
    pad = params["pad"]
    norm_indices = params["norm_indices"]
    align = params.get("align", "lmeds")
    hA, wA = imgA.shape[:2]

    hB, wB = imgB.shape[:2]
    if hB != hA or wB != wA:
        sx, sy = wA / wB, hA / hB
        imgB = cv2.resize(imgB, (wA, hA), interpolation=cv2.INTER_LINEAR)
        ptsB = ptsB * np.array([sx, sy], dtype=np.float32)

    _imgA = cv2.copyMakeBorder(imgA, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    _imgB = cv2.copyMakeBorder(imgB, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    _ptsA, _ptsB = ptsA + pad, ptsB + pad

    _imgB, _ptsB = normalize_to_reference(_imgB, _ptsB, _ptsA, norm_indices,
                                          align=align)
    h, w = _imgA.shape[:2]
    steps = params["steps"]

    frames = []
    for step in range(steps):
        alpha = step / max(steps - 1, 1)
        morphed, pts_morph = morph_frame(_imgA, _ptsA, _imgB, _ptsB,
                                         tris, alpha, h, w)
        morphed = np.clip(morphed, 0, 255).astype(np.uint8)

        if params["mask_output"] and params["face_poly"] is not None:
            outer = make_face_mask(h, w, pts_morph,
                                   params["face_poly"],
                                   feather=params["feather"])[..., None]
            bg = np.full_like(morphed, params.get("bg_color", (128, 128, 128)),
                              dtype=np.float32)
            morphed = np.clip(
                morphed.astype(np.float32) * outer + bg * (1.0 - outer),
                0, 255).astype(np.uint8)

        morphed = morphed[pad:-pad, pad:-pad]
        pts_unpad = pts_morph - pad
        morphed = blur_lip_seam(morphed, pts_unpad, LIP_SEAM_INDICES,
                                thickness=10, blur_ksize=23)
        frames.append((alpha, morphed, pts_unpad))
    return frames


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STREAMLIT UI                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(page_title="Face Morphing Toolbox", layout="wide")
st.title("Face Morphing Toolbox")
st.caption(f"Version {APP_VERSION}")

with st.expander("Which tab should I use?"):
    st.markdown("""
| Tab | What it does | You need |
|-----|-------------|----------|
| **Transform** | Apply a shape/color/texture delta from one face onto another face (C + strength × (B − A)). Single output per B face. | A (baseline avg), C (target avg), B faces |
| **Transform Sequence** | Same as Transform but outputs a sequence of frames stepping from 0 to max strength. | Same as Transform |
| **Morph Two Faces** | Morph face A toward face B — A is both the baseline and target. | A (base face), B faces |
| **Morph Sequence** | Same as Morph Two Faces but outputs a sequence of frames. | Same as Morph Two Faces |
| **Average** | Compute the average face from a set of faces. All faces are warped to the mean shape and pixel-averaged. | Face images + .tem files |
""")
tab_transform, tab_transform_seq, tab_morph_two, tab_morph_seq, tab_average = st.tabs(
    ["Transform", "Transform Sequence", "Morph Two Faces", "Morph Sequence", "Average"])


# ── Sidebar: shared support files & settings ──────────────────────────────────

with st.sidebar:
    st.header("Support Files")
    tris_file = st.file_uploader("Triangulation (.npy)", type=["npy"], key="tris",
                                 help="Optional — built-in canonical triangulation "
                                      "used if omitted.")
    mask_file = st.file_uploader("Face mask (.msk)", type=["msk"], key="mask",
                                 help="Optional — built-in CFD face-only mask "
                                      "used if omitted.")
    sym_file = st.file_uploader("Symmetry map (.sym)", type=["sym"], key="sym",
                                help="Optional — needed for symmetrize.")

    st.divider()
    st.header("Shared Settings")
    feather = st.slider("Feather (mask blur)", 1, 50, 5)
    pad = st.slider("Padding (px)", 0, 200, 100, 10)
    mask_output = st.checkbox("Mask output to face polygon", value=True)
    bg_hex = st.color_picker(
        "Background color (cropped area)", value="#808080",
        help="Color composited behind the face when masking is on.")
    bg_color = tuple(int(bg_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    keep_full = st.checkbox(
        "Don't crop output (keep full image)", value=False,
        help="When checked, the full warped image is returned uncropped — "
             "no face-polygon masking or background fill is applied.")
    # A single switch the tabs can pass straight through: cropping/masking is
    # only active when the user both wants masking and hasn't asked for the
    # full uncropped image.
    mask_output = mask_output and not keep_full

    st.markdown("**Color / texture anchoring**")
    _anchor_labels = {
        "Subject (0 = subject, increase → target)": "subject",
        "Target (0 = target avg, increase → subject)": "target",
        "Residual (target avg + strength × (subject − from avg))": "residual",
    }
    _color_choice = st.selectbox(
        "Color anchor", list(_anchor_labels.keys()), index=0,
        help="How the color strength is interpreted. 'Residual' plants the "
             "subject's deviation from the FROM-average onto the TARGET average "
             "(0 = target-average color, 1 = subject's individuating color on "
             "the target base).")
    color_anchor = _anchor_labels[_color_choice]
    _tex_choice = st.selectbox(
        "Texture anchor", list(_anchor_labels.keys()), index=0,
        help="Same interpretation as the color anchor, applied to the "
             "high-frequency texture transfer.")
    texture_anchor = _anchor_labels[_tex_choice]
    chroma_boost = st.slider(
        "Skin saturation", 0.5, 2.0, 1.0, 0.05,
        help="Only applies when Color space is Lab. "
             "Lab a*/b* multiplier inside the face. 1.0 = no change; >1 "
             "counteracts an ashen/gray look (CFD averages are desaturated, so "
             "morphs built from them can look flat). ~1.2-1.3 is a typical lift.")

    expected_landmarks = st.number_input("Expected landmarks", value=189,
                                         min_value=1)


def zip_with_settings(file_dict, tab, key_prefix):
    """make_zip plus a settings.json recording version and all settings.

    Captures the shared sidebar settings and every widget of this tab
    (session_state keys starting with `key_prefix`); uploads are recorded
    by file name only.
    """
    def _plain(v):
        if hasattr(v, "name") and hasattr(v, "getvalue"):
            return v.name
        if isinstance(v, (list, tuple)):
            return [_plain(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    tab_widgets = {k[len(key_prefix):]: _plain(v)
                   for k, v in sorted(st.session_state.items())
                   if isinstance(k, str) and k.startswith(key_prefix)
                   and not k.endswith("_run")}
    settings = {
        "app": "Face Morphing Toolbox",
        "version": APP_VERSION,
        "source_sha256": APP_SOURCE_SHA256,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                   .isoformat(timespec="seconds"),
        "tab": tab,
        "alignment": {"method": "umeyama", "anchor_landmarks": [0, 1, 96]},
        "edge_fill": EDGE_FILL,
        "shared": {
            "triangulation": tris_file.name if tris_file else "built-in",
            "face_mask": mask_file.name if mask_file else "built-in",
            "symmetry_map": sym_file.name if sym_file else None,
            "feather": feather, "padding": pad,
            "mask_output": mask_output, "keep_full": keep_full,
            "background_rgb": list(bg_color),
            "color_anchor": color_anchor, "texture_anchor": texture_anchor,
            "skin_saturation": chroma_boost,
            "expected_landmarks": int(expected_landmarks),
        },
        "tab_settings": tab_widgets,
    }
    files = dict(file_dict)
    files["settings.json"] = json.dumps(settings, indent=2).encode()
    return make_zip(files)


def _load_shared():
    """Load tris, face_poly, sym_map from sidebar uploads."""
    tris = (np.load(io.BytesIO(tris_file.read())) if tris_file
            else _load_builtin_tris())
    if tris_file: tris_file.seek(0)
    face_poly = (load_msk(mask_file.read()) if mask_file
                 else BUILTIN_FACE_POLY)
    if mask_file: mask_file.seek(0)
    sym_map = None
    if sym_file:
        sym_map = load_sym(sym_file.read()); sym_file.seek(0)
    return tris, face_poly, sym_map


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1:  TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════

with tab_transform:
    st.subheader("Parameters")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        shape_strength = st.slider("Shape strength", -2.0, 2.0, 1.0, 0.05,
                                   key="t_shape")
    with tc2:
        # Residual anchor: 0 = pure target-average tone, 1 = full individual
        # deviation (too dark/gray for dark->light). ~0.3 keeps the light/warm
        # target tone with a touch of the person's own coloring.
        color_strength = st.slider("Color strength", -2.0, 2.0, 0.3, 0.05,
                                   key="t_color")
    with tc3:
        texture_strength = st.slider("Texture strength", -2.0, 2.0, 0.2, 0.05,
                                     key="t_texture")

    tc4, tc5, tc6 = st.columns(3)
    with tc4:
        texture_blur_sigma = st.slider("Texture blur sigma", 5, 80, 10,
                                       key="t_blur")
    with tc5:
        t_cs = st.selectbox("Color space", ["RGB", "Lab", "YUV"], key="t_cs",
                            help="RGB (default): additive transfer in RGB, keeps skin warm. Lab: perceptually uniform; enables the Skin saturation slider. YUV: legacy.")
    with tc6:
        symmetrize_b = st.checkbox("Symmetrize B faces", value=False,
                                   key="t_sym",
                                   help="Requires .sym file in sidebar.")

    tc7, tc8, tc9 = st.columns(3)
    with tc7:
        of_crop = st.checkbox("OpenFace crop (flat forehead)", value=False,
                              key="t_ofcrop",
                              help="Square 512-style crop: jaw, eyes and brows "
                                   "fit to a fixed template, like "
                                   "OpenFace aligned output.")
    with tc8:
        of_mask = st.checkbox("Mask to face (black)", value=True,
                              key="t_ofmask",
                              help="Black out everything outside the jaw "
                                   "and raised-brow outline.")
    with tc9:
        of_size = st.selectbox("Crop size (px)", [256, 512, 1024], index=1,
                               key="t_ofsize")

    st.subheader("Images")
    col_a, col_c = st.columns(2)
    with col_a:
        st.markdown("**A — Baseline average**  *(image + .tem)*")
        img_a_file, tem_a_file = single_face_upload(st.file_uploader(
            "A image + TEM", type=["jpg", "jpeg", "png", "tem"], accept_multiple_files=True,
            key="t_A"), expected_landmarks)
        _show_preview(img_a_file)
    with col_c:
        st.markdown("**C — Transform target**  *(image + .tem)*")
        img_c_file, tem_c_file = single_face_upload(st.file_uploader(
            "C image + TEM", type=["jpg", "jpeg", "png", "tem"], accept_multiple_files=True,
            key="t_C"), expected_landmarks)
        _show_preview(img_c_file)

    st.markdown("**B — Individual face(s)**  *(upload .jpg + matching .tem for each)*")
    t_b_files = st.file_uploader("B images + TEMs", type=["jpg", "jpeg", "png", "tem"],
                                 accept_multiple_files=True, key="t_bfiles")

    col_suf, _ = st.columns([1, 3])
    with col_suf:
        t_suffix = st.text_input("Filename suffix", value="_asM", key="t_suffix")

    t_errors = []
    if not img_a_file:  t_errors.append("Upload A's image.")
    if not tem_a_file:  t_errors.append("Upload A's .tem.")
    if not img_c_file:  t_errors.append("Upload C's image.")
    if not tem_c_file:  t_errors.append("Upload C's .tem.")
    if not t_b_files:   t_errors.append("Upload B face files (.jpg + .tem).")
    if symmetrize_b and not sym_file:
        t_errors.append("Upload a .sym file (sidebar) for symmetrization.")

    if t_errors:
        for e in t_errors:
            st.caption(f"⚠️ {e}")
    else:
        paired = pair_uploaded_files(t_b_files)
        st.info(f"Found **{len(paired)}** matched B pairs → "
                f"**{len(paired)}** output images")
        if st.button("Run Transform", type="primary", key="t_run"):
            imgA = read_image(img_a_file)
            ptsA, _ = load_tem_from_bytes(tem_a_file.read(), expected_landmarks)
            tem_a_file.seek(0)
            imgC = read_image(img_c_file)
            ptsC, curve_block = load_tem_from_bytes(tem_c_file.read(), expected_landmarks)
            tem_c_file.seek(0)
            tris, face_poly, sym_map = _load_shared()
            params = dict(shape=shape_strength, color=color_strength,
                          texture=texture_strength, feather=feather,
                          blur_sigma=texture_blur_sigma, pad=pad,
                          mask_output=mask_output, bg_color=bg_color, color_anchor=color_anchor, texture_anchor=texture_anchor, chroma_boost=chroma_boost, norm_indices=(0, 1, 96), align="umeyama", edge_fill=EDGE_FILL,
                          face_poly=face_poly, use_yuv=False, color_space=t_cs.lower(),
                          of_crop=of_crop, of_mask=of_mask, of_size=of_size)

            results, zip_files, skipped = [], {}, []
            progress = st.progress(0, text="Starting…")
            for idx, (stem, img_f, tem_f) in enumerate(paired):
                progress.progress(idx / len(paired), text=f"Processing {stem}")
                try:
                    imgB = read_image(img_f)
                    ptsB, _ = load_tem_from_bytes(tem_f.read(), expected_landmarks)
                    tem_f.seek(0)
                except Exception as exc:
                    skipped.append((stem, str(exc))); continue
                if symmetrize_b and sym_map is not None:
                    imgB, ptsB = symmetrize_image(imgB, ptsB, sym_map, tris)
                morphed, target_pts = run_transform_single(
                    imgA, ptsA, imgB, ptsB, imgC, ptsC, curve_block, tris, params)
                out_name = f"{stem}{t_suffix}"
                zip_files[f"{out_name}.jpg"] = encode_jpg(morphed)
                zip_files[f"{out_name}.tem"] = save_tem_to_bytes(target_pts, curve_block)
                results.append((stem, morphed))
            progress.progress(1.0, text="Done!")
            for i in range(0, len(results), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        with col:
                            st.image(results[i+j][1], caption=f"{results[i+j][0]}{t_suffix}", clamp=True)
            if skipped:
                with st.expander(f"{len(skipped)} skipped"):
                    for s, r in skipped: st.text(f"{s}: {r}")
            st.download_button("Download results (.zip)", zip_with_settings(zip_files, "Transform", "t_"),
                               "transform_results.zip", "application/zip")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2:  TRANSFORM SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_transform_seq:
    st.caption("Like Transform, but steps from 0 to max strength across N frames.")
    st.subheader("Parameters")
    ts0, _ = st.columns([1, 3])
    with ts0:
        ts_steps = st.slider("Number of frames", 2, 30, 6, key="ts_steps")
    ts_c1, ts_c2, ts_c3 = st.columns(3)
    with ts_c1:
        ts_shape = st.slider("Max shape strength", -2.0, 2.0, 1.0, 0.05, key="ts_shape")
    with ts_c2:
        ts_color = st.slider("Max color strength", -2.0, 2.0, 0.5, 0.05, key="ts_color")
    with ts_c3:
        ts_texture = st.slider("Max texture strength", -2.0, 2.0, 0.5, 0.05, key="ts_texture")
    ts_c4, ts_c5, ts_c6 = st.columns(3)
    with ts_c4:
        ts_blur = st.slider("Texture blur sigma", 5, 80, 10, key="ts_blur")
    with ts_c5:
        ts_cs = st.selectbox("Color space", ["RGB", "Lab", "YUV"], key="ts_cs",
                             help="RGB (default): additive transfer in RGB, keeps skin warm. Lab: perceptually uniform; enables the Skin saturation slider. YUV: legacy.")
    with ts_c6:
        ts_sym = st.checkbox("Symmetrize B faces", value=False, key="ts_sym")

    st.subheader("Images")
    ts_a, ts_c = st.columns(2)
    with ts_a:
        st.markdown("**A — Baseline average**  *(image + .tem)*")
        ts_img_a, ts_tem_a = single_face_upload(st.file_uploader(
            "A image + TEM", type=["jpg", "jpeg", "png", "tem"], accept_multiple_files=True,
            key="ts_A"), expected_landmarks)
        _show_preview(ts_img_a)
    with ts_c:
        st.markdown("**C — Transform target**  *(image + .tem)*")
        ts_img_c, ts_tem_c = single_face_upload(st.file_uploader(
            "C image + TEM", type=["jpg", "jpeg", "png", "tem"], accept_multiple_files=True,
            key="ts_C"), expected_landmarks)
        _show_preview(ts_img_c)
    st.markdown("**B — Individual face(s)**")
    ts_b_files = st.file_uploader("B images + TEMs", type=["jpg", "jpeg", "png", "tem"],
                                  accept_multiple_files=True, key="ts_bfiles")

    ts_errors = []
    if not ts_img_a:    ts_errors.append("Upload A's image.")
    if not ts_tem_a:    ts_errors.append("Upload A's .tem.")
    if not ts_img_c:    ts_errors.append("Upload C's image.")
    if not ts_tem_c:    ts_errors.append("Upload C's .tem.")
    if not ts_b_files:  ts_errors.append("Upload B face files.")
    if ts_sym and not sym_file: ts_errors.append("Upload .sym (sidebar).")

    if ts_errors:
        for e in ts_errors: st.caption(f"⚠️ {e}")
    else:
        paired = pair_uploaded_files(ts_b_files)
        st.info(f"Ready: **{len(paired)}** faces × **{ts_steps}** frames = "
                f"**{len(paired) * ts_steps}** output images")
        if st.button("Generate Transform Sequences", type="primary", key="ts_run"):
            imgA = read_image(ts_img_a)
            ptsA, _ = load_tem_from_bytes(ts_tem_a.read(), expected_landmarks); ts_tem_a.seek(0)
            imgC = read_image(ts_img_c)
            ptsC, curve_block = load_tem_from_bytes(ts_tem_c.read(), expected_landmarks); ts_tem_c.seek(0)
            tris, face_poly, sym_map = _load_shared()

            zip_files, all_seqs, skipped = {}, [], []
            progress = st.progress(0, text="Starting…")
            total = len(paired) * ts_steps; done = 0
            for idx, (stem, img_f, tem_f) in enumerate(paired):
                try:
                    imgB = read_image(img_f)
                    ptsB, _ = load_tem_from_bytes(tem_f.read(), expected_landmarks); tem_f.seek(0)
                except Exception as exc:
                    skipped.append((stem, str(exc))); done += ts_steps; continue
                if ts_sym and sym_map is not None:
                    imgB, ptsB = symmetrize_image(imgB, ptsB, sym_map, tris)
                seq_frames = []
                for step in range(ts_steps):
                    t = step / max(ts_steps - 1, 1)
                    progress.progress(done / total, text=f"{stem} frame {step+1}/{ts_steps}")
                    params = dict(shape=t*ts_shape, color=t*ts_color, texture=t*ts_texture,
                                  feather=feather, blur_sigma=ts_blur, pad=pad,
                                  mask_output=mask_output, bg_color=bg_color, color_anchor=color_anchor, texture_anchor=texture_anchor, chroma_boost=chroma_boost, norm_indices=(0, 1, 96), align="umeyama", edge_fill=EDGE_FILL,
                                  face_poly=face_poly, use_yuv=False, color_space=ts_cs.lower())
                    morphed, tpts = run_transform_single(imgA, ptsA, imgB, ptsB, imgC, ptsC,
                                                         curve_block, tris, params)
                    tag = f"{int(t*100):03d}"
                    zip_files[f"{stem}/{stem}_transform_{tag}.jpg"] = encode_jpg(morphed)
                    zip_files[f"{stem}/{stem}_transform_{tag}.tem"] = save_tem_to_bytes(tpts, curve_block)
                    seq_frames.append((t, morphed))
                    done += 1
                all_seqs.append((stem, [seq_frames[0], seq_frames[len(seq_frames)//2], seq_frames[-1]]))
            progress.progress(1.0, text="Done!")
            for stem, previews in all_seqs:
                st.markdown(f"**{stem}**")
                cols = st.columns(len(previews))
                for col, (t, img) in zip(cols, previews):
                    with col: st.image(img, caption=f"{int(t*100)}%", clamp=True)
            if skipped:
                with st.expander(f"{len(skipped)} skipped"):
                    for s, r in skipped: st.text(f"{s}: {r}")
            st.download_button("Download results (.zip)", zip_with_settings(zip_files, "Transform Sequence", "ts_"),
                               "transform_seq_results.zip", "application/zip")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3:  MORPH TWO FACES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_morph_two:
    st.caption("Morph face A toward face B — A is both the baseline and target.")
    st.subheader("Parameters")
    m2c1, m2c2, m2c3 = st.columns(3)
    with m2c1: m2_shape = st.slider("Shape strength", -2.0, 2.0, 0.5, 0.05, key="m2_shape")
    with m2c2: m2_color = st.slider("Color strength", -2.0, 2.0, 0.5, 0.05, key="m2_color")
    with m2c3: m2_texture = st.slider("Texture strength", -2.0, 2.0, 0.5, 0.05, key="m2_texture")
    m2c4, m2c5, m2c6 = st.columns(3)
    with m2c4: m2_blur = st.slider("Texture blur sigma", 5, 80, 30, key="m2_blur")
    with m2c5: m2_cs = st.selectbox("Color space", ["RGB", "Lab", "YUV"], key="m2_cs",
                                    help="RGB (default): additive transfer in RGB, keeps skin warm. Lab: perceptually uniform; enables the Skin saturation slider. YUV: legacy.")
    with m2c6: m2_sym = st.checkbox("Symmetrize B faces", value=False, key="m2_sym")

    st.subheader("Images")
    st.markdown("**A — Base face**  *(image + .tem)*")
    m2_img_a, m2_tem_a = single_face_upload(st.file_uploader(
        "A image + TEM", type=["jpg", "jpeg", "png", "tem"], accept_multiple_files=True,
        key="m2_A"), expected_landmarks)
    _show_preview(m2_img_a)
    st.markdown("**B — Face(s) to morph toward**")
    m2_b_files = st.file_uploader("B images + TEMs", type=["jpg", "jpeg", "png", "tem"],
                                  accept_multiple_files=True, key="m2_bfiles")
    m2_col_suf, _ = st.columns([1, 3])
    with m2_col_suf:
        m2_suffix = st.text_input("Filename suffix", value="_morphed", key="m2_suffix")

    m2_errors = []
    if not m2_img_a:   m2_errors.append("Upload A's image.")
    if not m2_tem_a:   m2_errors.append("Upload A's .tem.")
    if not m2_b_files: m2_errors.append("Upload B face files.")
    if m2_sym and not sym_file: m2_errors.append("Upload .sym (sidebar).")
    if m2_errors:
        for e in m2_errors: st.caption(f"⚠️ {e}")
    else:
        paired = pair_uploaded_files(m2_b_files)
        st.info(f"Ready: **{len(paired)}** faces → **{len(paired)}** output images")
        if st.button("Run Morph", type="primary", key="m2_run"):
            imgA = read_image(m2_img_a)
            ptsA, curve_block = load_tem_from_bytes(m2_tem_a.read(), expected_landmarks); m2_tem_a.seek(0)
            tris, face_poly, sym_map = _load_shared()
            params = dict(shape=m2_shape, color=m2_color, texture=m2_texture,
                          feather=feather, blur_sigma=m2_blur, pad=pad,
                          mask_output=mask_output, bg_color=bg_color, color_anchor=color_anchor, texture_anchor=texture_anchor, chroma_boost=chroma_boost, norm_indices=(0, 1, 96), align="umeyama", edge_fill=EDGE_FILL,
                          face_poly=face_poly, use_yuv=False, color_space=m2_cs.lower())
            results, zip_files, skipped = [], {}, []
            progress = st.progress(0, text="Starting…")
            for idx, (stem, img_f, tem_f) in enumerate(paired):
                progress.progress(idx / len(paired), text=f"Processing {stem}")
                try:
                    imgB = read_image(img_f)
                    ptsB, _ = load_tem_from_bytes(tem_f.read(), expected_landmarks); tem_f.seek(0)
                except Exception as exc: skipped.append((stem, str(exc))); continue
                if m2_sym and sym_map is not None:
                    imgB, ptsB = symmetrize_image(imgB, ptsB, sym_map, tris)
                morphed, tpts = run_transform_single(imgA, ptsA, imgB, ptsB, imgA, ptsA,
                                                      curve_block, tris, params)
                zip_files[f"{stem}{m2_suffix}.jpg"] = encode_jpg(morphed)
                zip_files[f"{stem}{m2_suffix}.tem"] = save_tem_to_bytes(tpts, curve_block)
                results.append((stem, morphed))
            progress.progress(1.0, text="Done!")
            for i in range(0, len(results), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i+j < len(results):
                        with col: st.image(results[i+j][1], caption=f"{results[i+j][0]}{m2_suffix}", clamp=True)
            if skipped:
                with st.expander(f"{len(skipped)} skipped"):
                    for s, r in skipped: st.text(f"{s}: {r}")
            st.download_button("Download results (.zip)", zip_with_settings(zip_files, "Morph Two Faces", "m2_"),
                               "morph_results.zip", "application/zip")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4:  MORPH SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_morph_seq:
    st.caption("Like Morph Two Faces but steps from 0 to max strength across N frames.")
    st.subheader("Parameters")
    ms0, _ = st.columns([1, 3])
    with ms0: ms_steps = st.slider("Number of frames", 2, 30, 6, key="ms_steps")
    ms_c1, ms_c2, ms_c3 = st.columns(3)
    with ms_c1: ms_shape = st.slider("Max shape strength", -2.0, 2.0, 1.0, 0.05, key="ms_shape")
    with ms_c2: ms_color = st.slider("Max color strength", -2.0, 2.0, 0.5, 0.05, key="ms_color")
    with ms_c3: ms_texture = st.slider("Max texture strength", -2.0, 2.0, 0.5, 0.05, key="ms_texture")
    ms_c4, ms_c5, ms_c6 = st.columns(3)
    with ms_c4: ms_blur = st.slider("Texture blur sigma", 5, 80, 30, key="ms_blur")
    with ms_c5: ms_cs = st.selectbox("Color space", ["RGB", "Lab", "YUV"], key="ms_cs",
                                     help="RGB (default): additive transfer in RGB, keeps skin warm. Lab: perceptually uniform; enables the Skin saturation slider. YUV: legacy.")
    with ms_c6: ms_sym = st.checkbox("Symmetrize B faces", value=False, key="ms_sym")

    st.subheader("Images")
    st.markdown("**A — Base face**  *(image + .tem)*")
    ms_img_a, ms_tem_a = single_face_upload(st.file_uploader(
        "A image + TEM", type=["jpg", "jpeg", "png", "tem"], accept_multiple_files=True,
        key="ms_A"), expected_landmarks)
    _show_preview(ms_img_a)
    st.markdown("**B — Target face(s)**")
    ms_b_files = st.file_uploader("B images + TEMs", type=["jpg", "jpeg", "png", "tem"],
                                  accept_multiple_files=True, key="ms_bfiles")

    ms_errors = []
    if not ms_img_a:   ms_errors.append("Upload A's image.")
    if not ms_tem_a:   ms_errors.append("Upload A's .tem.")
    if not ms_b_files: ms_errors.append("Upload B face files.")
    if ms_sym and not sym_file: ms_errors.append("Upload .sym (sidebar).")
    if ms_errors:
        for e in ms_errors: st.caption(f"⚠️ {e}")
    else:
        paired = pair_uploaded_files(ms_b_files)
        st.info(f"Ready: **{len(paired)}** faces × **{ms_steps}** frames = "
                f"**{len(paired) * ms_steps}** output images")
        if st.button("Generate Morph Sequences", type="primary", key="ms_run"):
            imgA = read_image(ms_img_a)
            ptsA, curve_block = load_tem_from_bytes(ms_tem_a.read(), expected_landmarks); ms_tem_a.seek(0)
            tris, face_poly, sym_map = _load_shared()
            zip_files, all_seqs, skipped = {}, [], []
            progress = st.progress(0, text="Starting…")
            total = len(paired) * ms_steps; done = 0
            for idx, (stem, img_f, tem_f) in enumerate(paired):
                try:
                    imgB = read_image(img_f)
                    ptsB, _ = load_tem_from_bytes(tem_f.read(), expected_landmarks); tem_f.seek(0)
                except Exception as exc: skipped.append((stem, str(exc))); done += ms_steps; continue
                if ms_sym and sym_map is not None:
                    imgB, ptsB = symmetrize_image(imgB, ptsB, sym_map, tris)
                seq_frames = []
                for step in range(ms_steps):
                    t = step / max(ms_steps - 1, 1)
                    progress.progress(done / total, text=f"{stem} frame {step+1}/{ms_steps}")
                    params = dict(shape=t*ms_shape, color=t*ms_color, texture=t*ms_texture,
                                  feather=feather, blur_sigma=ms_blur, pad=pad,
                                  mask_output=mask_output, bg_color=bg_color, color_anchor=color_anchor, texture_anchor=texture_anchor, chroma_boost=chroma_boost, norm_indices=(0, 1, 96), align="umeyama", edge_fill=EDGE_FILL,
                                  face_poly=face_poly, use_yuv=False, color_space=ms_cs.lower())
                    morphed, tpts = run_transform_single(imgA, ptsA, imgB, ptsB, imgA, ptsA,
                                                          curve_block, tris, params)
                    tag = f"{int(t*100):03d}"
                    zip_files[f"{stem}/{stem}_morph_{tag}.jpg"] = encode_jpg(morphed)
                    zip_files[f"{stem}/{stem}_morph_{tag}.tem"] = save_tem_to_bytes(tpts, curve_block)
                    seq_frames.append((t, morphed)); done += 1
                all_seqs.append((stem, [seq_frames[0], seq_frames[len(seq_frames)//2], seq_frames[-1]]))
            progress.progress(1.0, text="Done!")
            for stem, previews in all_seqs:
                st.markdown(f"**{stem}**")
                cols = st.columns(len(previews))
                for col, (t, img) in zip(cols, previews):
                    with col: st.image(img, caption=f"{int(t*100)}%", clamp=True)
            if skipped:
                with st.expander(f"{len(skipped)} skipped"):
                    for s, r in skipped: st.text(f"{s}: {r}")
            st.download_button("Download results (.zip)", zip_with_settings(zip_files, "Morph Sequence", "ms_"),
                               "morph_seq_results.zip", "application/zip")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5:  DOPPELGANGER SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6:  JITTER
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7:  RC STIMULI
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 8:  AVERAGE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_average:
    st.caption(
        "Compute the average face from a set of face images. "
        "All faces are normalized, warped to the mean shape, "
        "and pixel-averaged."
    )

    st.subheader("Images")
    st.markdown("Upload all face images + matching .tem files together")
    avg_files = st.file_uploader(
        "Face images + TEMs",
        type=["jpg", "jpeg", "png", "tem"],
        accept_multiple_files=True,
        key="avg_files",
    )

    st.subheader("Output")
    avg_col1, avg_col2 = st.columns([2, 1])
    with avg_col1:
        avg_filename = st.text_input("Output filename", value="average",
                                     key="avg_name",
                                     help="The average will be saved as "
                                          "{filename}.jpg + .tem in the zip.")

    avg_errors = []
    if not avg_files:
        avg_errors.append("Upload face files (.jpg + .tem).")

    if avg_errors:
        for e in avg_errors:
            st.caption(f"⚠️ {e}")
    else:
        paired = pair_uploaded_files(avg_files)
        st.info(f"Found **{len(paired)}** matched face pairs → "
                f"will compute average from {len(paired)} faces")

        if len(paired) < 2:
            st.warning("Need at least 2 faces to compute an average.")
        elif st.button("Compute Average", type="primary", key="avg_run"):
            tris, face_poly, _ = _load_shared()

            images, all_pts = [], []
            skipped = []
            progress = st.progress(0, text="Loading faces…")

            for idx, (stem, img_f, tem_f) in enumerate(paired):
                progress.progress(idx / len(paired),
                                  text=f"Loading {stem}")
                try:
                    img = read_image(img_f)
                    pts, _ = load_tem_from_bytes(tem_f.read(),
                                                expected_landmarks)
                    tem_f.seek(0)
                    images.append(img)
                    all_pts.append(pts)
                except Exception as exc:
                    skipped.append((stem, str(exc)))

            if len(images) < 2:
                st.error("Not enough valid faces to average "
                         f"({len(images)} loaded, {len(skipped)} skipped).")
            else:
                progress.progress(0.5, text=f"Averaging {len(images)} faces…")

                avg_img, avg_pts = average_faces(
                    images, all_pts, tris,
                    face_poly=face_poly,
                    feather=feather,
                    pad=pad,
                    mask_output=mask_output,
                    bg_color=bg_color,
                )

                progress.progress(1.0, text="Done!")

                st.subheader("Result")
                st.image(avg_img, caption=f"Average of {len(images)} faces",
                         clamp=True)

                if skipped:
                    with st.expander(f"{len(skipped)} skipped"):
                        for s, r in skipped:
                            st.text(f"{s}: {r}")

                # Build download
                zip_files = {
                    f"{avg_filename}.jpg": encode_jpg(avg_img),
                    f"{avg_filename}.tem": save_tem_to_bytes(avg_pts),
                }
                st.download_button(
                    "Download average (.zip)",
                    zip_with_settings(zip_files, "Average", "avg_"),
                    f"{avg_filename}.zip",
                    "application/zip",
                )
