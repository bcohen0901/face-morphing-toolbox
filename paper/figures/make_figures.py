"""Build the manuscript figures from the Face Morphing Toolbox (v1.2.0).

Every image is generated here from CFD photographs, CFD templates, and the
CFD category averages, so the figures can be reproduced exactly.

    python make_figures.py          # writes fig_*.png (300 dpi) next to this file
"""

import glob
import os
import sys
import types
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
STRENGTH = 0.75  # color and texture strength used throughout (shape = 1)
GRAY = (128, 128, 128)
INK, MUTED, BLUE = "#0b0b0b", "#52514e", "#2a78d6"

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "savefig.dpi": 300,
    }
)

src = open(APP).read()
lib = types.ModuleType("toolbox")
lib.__file__ = APP
exec(compile(src[: src.find("st.set_page_config")], APP, "exec"), lib.__dict__)
TRIS = lib._load_builtin_tris()
POLY = lib.BUILTIN_FACE_POLY


def read(path):
    """Image (RGB, float32), landmarks, and template curves for the photo at `path` and its .tem file."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
    pts, curves = lib.load_tem(path[:-4] + ".tem")
    return img, pts, curves


def average(group):
    """The CFD category average for `group` (e.g. "AM")."""
    return read(f"{AVG}/CFD-{group}-AVG.jpg")


def original(ident):
    """The neutral CFD photograph for identity `ident` (e.g. "AM-204")."""
    group, num = ident.split("-")
    return read(glob.glob(f"{STIM}/{group}/CFD-{group}-{num}-*-N.jpg")[0])


def transform(A, B, C, shape=1.0, color=STRENGTH, texture=STRENGTH):
    """D = C + strength * (B - A); returns (image, landmarks)."""
    params = dict(
        shape=shape,
        color=color,
        texture=texture,
        feather=5,
        blur_sigma=10,
        pad=100,
        mask_output=True,
        bg_color=GRAY,
        color_anchor="subject",
        texture_anchor="subject",
        chroma_boost=1.0,
        norm_indices=(0, 1, 96),
        align="umeyama",
        face_poly=POLY,
        use_yuv=False,
        color_space="rgb",
        edge_fill=lib.EDGE_FILL,
    )
    img, pts = lib.run_transform_single(
        A[0], A[1], B[0], B[1], C[0], C[1], C[2], TRIS, params
    )
    return img, pts


def masked(face, ref=None):
    """A photograph or average masked to the face, matching the doppelgangers.

    With `ref` (an average), the face is first aligned to that average's frame,
    as the Toolbox does for its inputs, so sizes and positions are comparable.
    """
    img, pts, _ = face
    if ref is not None:
        img, pts = lib.normalize_to_reference(
            img, pts, ref[1], (0, 1, 96), align="umeyama"
        )
    h, w = img.shape[:2]
    m = lib.make_face_mask(h, w, pts, POLY, feather=5)[..., None]
    return (img * m + np.array(GRAY, np.float32) * (1 - m)).astype(np.uint8), pts


def blend_averages(A, C, lam):
    """Average lying a proportion lam of the way from A to C (Equation 4)."""
    imgC, ptsC = lib.normalize_to_reference(
        C[0], C[1], A[1], (0, 1, 96), align="umeyama"
    )
    h, w = A[0].shape[:2]
    img, pts = lib.morph_frame(A[0], A[1], imgC, ptsC, TRIS, lam, h, w)
    return img.astype(np.float32), pts.astype(np.float32), A[2]


def crop_box(pts_list, pad=0.12, all_points=False):
    """Square crop (x, y, side) around the face outline, or all landmarks, of every shape in `pts_list`."""
    allp = np.vstack([p if all_points else p[POLY] for p in pts_list])
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(cx - side / 2), int(cy - side / 2), int(side)


def crop(img, box):
    """Cut the square `box` from `img`, padding with gray where it extends past the image."""
    x, y, s = box
    h, w = img.shape[:2]
    canvas = np.full((s, s, 3), GRAY, np.uint8)
    xs, ys = max(x, 0), max(y, 0)
    xe, ye = min(x + s, w), min(y + s, h)
    canvas[ys - y : ye - y, xs - x : xe - x] = img[ys:ye, xs:xe]
    return canvas


def show(ax, img, title=None, sub=None):
    """Draw `img` on `ax` without axes, with an optional title above and caption below."""
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, color=INK, fontsize=8, pad=3)
    if sub:
        ax.text(
            0.5,
            -0.04,
            sub,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color=MUTED,
        )


def doppelgangers(ident):
    """Original (masked), same-category, and cross-category doppelganger."""
    group = ident.split("-")[0]
    other = ("W" if group[0] == "A" else "A") + group[1]
    B, A = original(ident), average(group)
    same = transform(A, B, A)
    cross = transform(A, B, average(other))
    return masked(B, A), same, cross


# ── Figure 1: example doppelganger pairs ──────────────────────────────────
def figure_examples(idents=("AM-204", "WF-020")):
    """Figure 1: original, same-category, and cross-category doppelganger for each identity."""
    label = {"A": "East Asian", "W": "White"}
    fig, axes = plt.subplots(len(idents), 3, figsize=(5.0, 1.85 * len(idents) + 0.3))
    for r, (row, ident) in enumerate(zip(axes, idents)):
        trio = doppelgangers(ident)
        box = crop_box([t[1] for t in trio])
        src_race, oth = label[ident[0]], label["W" if ident[0] == "A" else "A"]
        heads = [
            "Original photograph",
            "Same-category\ndoppelganger",
            "Cross-category\ndoppelganger",
        ]
        subs = [src_race, src_race, oth]
        for c, (ax, (img, _)) in enumerate(zip(row, trio)):
            show(ax, crop(img, box), heads[c] if r == 0 else None, subs[c])
        row[0].set_ylabel(f"CFD-{ident}", fontsize=8, color=INK)
    fig.tight_layout(w_pad=0.6, h_pad=1.2)
    fig.savefig(os.path.join(HERE, "fig_examples.png"))
    plt.close(fig)


# ── Equation figure: D = C + (B − A) ──────────────────────────────────────
def figure_equation(ident="AM-204"):
    """Figure 2: the doppelganger equation illustrated with the averages and the source face."""
    group = ident.split("-")[0]
    other = ("W" if group[0] == "A" else "A") + group[1]
    A, B, Cw = average(group), original(ident), average(other)
    same, cross = transform(A, B, A), transform(A, B, Cw)
    mB, mA, mCw = masked(B, A), masked(A), masked(Cw, A)
    box = crop_box([mB[1], mA[1], mCw[1], same[1], cross[1]])
    rows = [
        ("Same-category doppelganger", mA, same),
        ("Cross-category doppelganger", mCw, cross),
    ]
    fig = plt.figure(figsize=(6.8, 3.4))
    gs = fig.add_gridspec(
        2, 7, width_ratios=[1, 0.18, 1, 0.18, 1, 0.18, 1], wspace=0.05, hspace=0.35
    )
    heads = [
        "Source face (B)",
        "Source average (A)",
        "Target average (C)",
        "Doppelganger (D)",
    ]
    subs = [
        [f"CFD-{ident}", "East Asian male", "East Asian male", "East Asian"],
        [f"CFD-{ident}", "East Asian male", "White male", "White"],
    ]
    for r, (_, C, D) in enumerate(rows):
        imgs = [mB[0], mA[0], C[0], D[0]]
        for c, img in enumerate(imgs):
            ax = fig.add_subplot(gs[r, 2 * c])
            show(ax, crop(img, box), heads[c] if r == 0 else None, subs[r][c])
            if c == 3:
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(BLUE)
                    sp.set_linewidth(1.5)
        for c, sym in zip((1, 3, 5), ("−", "+", "=")):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            ax.text(0.5, 0.5, sym, ha="center", va="center", fontsize=16, color=INK)
    for r, (label, _, _) in enumerate(rows):
        fig.text(
            0.075,
            0.70 - r * 0.43,
            label.replace(" ", "\n", 1),
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
        )
    fig.text(
        0.5,
        0.005,
        "D = C + strength × (B − A). The source face’s deviation from its own "
        "category average is added to the target average.",
        ha="center",
        fontsize=7.5,
        color=MUTED,
    )
    fig.savefig(os.path.join(HERE, "fig_equation.png"), bbox_inches="tight")
    plt.close(fig)


# ── Figure X: step-by-step decomposition ──────────────────────────────────
def figure_steps(ident="AM-204"):
    """Figure 3: the target average, then shape, color, and texture added in turn."""
    group = ident.split("-")[0]
    other = ("W" if group[0] == "A" else "A") + group[1]
    A, B, C = average(group), original(ident), average(other)
    base, pts_c = transform(A, B, C, shape=0, color=0, texture=0)  # target average
    shape_only, pts_d = transform(A, B, C, shape=1, color=0, texture=0)
    with_color, _ = transform(A, B, C, shape=1, color=STRENGTH, texture=0)
    final, _ = transform(A, B, C)
    box = crop_box([pts_c, pts_d], pad=0.04, all_points=True)
    x, y, _ = box
    panels = [
        (
            "1–2. Shape",
            "Gray = target average\nBlue = doppelganger,\n"
            r"$x_D = x_C + \sigma(x_B - x_A)$",
            base,
        ),
        (
            "3. Warp",
            "Target average warped\nto the doppelganger\nshape (σ = 1)",
            shape_only,
        ),
        (
            "4. Color",
            f"Source face’s color\ndeviation added\n(κ = {STRENGTH:.2f})".replace(
                "0.", "."
            ),
            with_color,
        ),
        (
            "5–6. Texture, finishing",
            f"Fine texture added\n(τ = {STRENGTH:.2f})".replace("0.", ".")
            + " and face\nmasked onto background",
            final,
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.6))
    for ax, (title, sub, img) in zip(axes, panels):
        show(ax, crop(img, box), title, sub)
    for (xc, yc), (xd, yd) in zip(pts_c, pts_d):
        axes[0].plot([xc - x, xd - x], [yc - y, yd - y], color=BLUE, lw=0.6)
    axes[0].scatter(
        pts_c[:, 0] - x, pts_c[:, 1] - y, s=1.2, color="#d6d5d0", linewidths=0, zorder=3
    )
    axes[0].scatter(
        pts_d[:, 0] - x, pts_d[:, 1] - y, s=1.2, color=BLUE, linewidths=0, zorder=4
    )
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.88, bottom=0.25)
    fig.savefig(os.path.join(HERE, "fig_steps.png"), bbox_inches="tight")
    plt.close(fig)


# ── Continuum figure ──────────────────────────────────────────────────────
def figure_continuum(ident="AM-204", steps=11):
    """Figure 4: the continuum from the same-category to the cross-category doppelganger (Equation 4)."""
    group = ident.split("-")[0]
    other = ("W" if group[0] == "A" else "A") + group[1]
    A, B, C = average(group), original(ident), average(other)
    frames = []
    for lam in np.linspace(0, 1, steps):
        frames.append(transform(A, B, blend_averages(A, C, lam)))
    box = crop_box([f[1] for f in frames])
    fig, axes = plt.subplots(1, steps, figsize=(6.8, 1.15))
    for ax, lam, (img, _) in zip(axes, np.linspace(0, 1, steps), frames):
        show(ax, crop(img, box), sub=f"λ = {lam:.1f}")
    axes[0].set_title("Same-category\n(East Asian)", fontsize=7, color=INK)
    axes[-1].set_title("Cross-category\n(White)", fontsize=7, color=INK)
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.78, bottom=0.12)
    fig.savefig(os.path.join(HERE, "fig_continuum.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_examples()
    figure_equation()
    figure_steps()
    figure_continuum()
    print("wrote", sorted(f for f in os.listdir(HERE) if f.endswith(".png")))
