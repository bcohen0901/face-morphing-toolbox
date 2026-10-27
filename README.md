# Face Morphing Toolbox

A browser-based tool for transforming, morphing, and averaging face images
using 189-point landmark templates (the Psychomorph/WebMorph `.tem` format).
It implements the doppelganger method described in Cohen, Lakshmi, and Correll
(in revision), which changes a face's apparent category (for example, race or
gender) while holding constant its individuating deviation from its own
category average.

- **Run it online.** https://huggingface.co/spaces/bcohen0901/face-morphing-share
- **Version.** 1.2.0 (see [CHANGELOG.md](CHANGELOG.md))
- **License.** MIT

## What it does

| Tab | Output |
|-----|--------|
| Transform | One image per source face, computed as C + strength × (B − A) |
| Transform Sequence | The same, stepped from 0 to a maximum strength over N frames |
| Morph Two Faces | Face A morphed toward face B |
| Morph Sequence | Frames from A to B |
| Average | The average of a set of faces (aligned, warped to the mean shape, pixel-averaged) |

In the Transform tabs, **A** is the source average (the average of the source
face's own category), **B** is the source face (several can be uploaded at
once), and **C** is the target average. With C = A the result is a
same-category doppelganger, and with C set to another category's average it is
a cross-category doppelganger. In the Morph tabs, A and B are the start and end
faces.

The Toolbox does not place landmarks. Every image needs a matching 189-point
`.tem` file, uploaded in the same box and paired by file name. Templates for
the Chicago Face Database are available from Singh, Gambrell, and Correll
(2022). Templates for other faces can be made in Psychomorph or WebMorph.

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| Shape strength | 1.00 | Proportion of the source face's shape deviation carried over |
| Color strength | 0.30 | Proportion of the source face's color deviation carried over |
| Texture strength | 0.20 | Amount of the source face's fine texture added |
| Texture blur sigma | 10 px | Separates fine texture from broader shading |
| Color space | RGB | RGB, Lab, or YUV. Lab enables the skin-saturation adjustment |
| Color / texture anchor | Subject | How strengths are applied (see the in-app help) |
| Feather | 5 px | Softness of the face-mask edge |
| Mask output | On, gray background | Composites the face onto a uniform background |
| OpenFace crop | Off | Optional standardized square crop |

Edge fill (1% of face width) is always on in the app. It fills color and
texture in a narrow band inside the face outline from the adjacent skin, so
hair or background at the outline cannot tint the rim.

The manuscript uses shape strength 1 and color and texture strengths of .75,
and describes how to choose values for a new stimulus set.

## Run it locally

With Python 3.12:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Or with Docker:

```bash
docker build -t face-morphing-toolbox .
docker run -p 7860:7860 face-morphing-toolbox
```

Then open http://localhost:7860.

## Reproducibility

Every download includes a `settings.json` file recording the app version, a
SHA-256 hash of the app source, the alignment method, every setting, and the
names of the input files. Given the same inputs, settings, and version, the
Toolbox produces identical output. When reporting stimuli, give the version and
the settings in that file, how the averages were built, and the source of the
landmark templates.

## Reproducing the manuscript

The [paper](paper) folder contains the scripts that generate Figures 1–5 and
the embedding analyses, and the results files they produced. See
[paper/README.md](paper/README.md).

## Citation

See [CITATION.cff](CITATION.cff). [Zenodo DOI to be added]
