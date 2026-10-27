# Changelog

## 1.2.0 (2026-10-26)

- Edge fill. Pixels at the very edge of the face in the source photograph and
  the averages blend in hair or background, which could leave a light rim
  around transformed faces. Color and texture in a narrow band inside the face
  outline (1% of face width) are now filled in from the adjacent skin. The
  setting is recorded in `settings.json`.

## 1.1.0 (2026-10-26)

- All transform and morph tabs use the least-squares (Umeyama) similarity
  alignment. They previously fell back to LMEDS.
- The "Use YUV" checkbox is replaced by a Color space menu (RGB default, Lab,
  YUV). The Skin saturation slider takes effect in Lab.
- Every download includes `settings.json` with the version, a SHA-256 hash of
  the app source, the alignment, all settings, and the input file names. The
  version is shown under the app title.
