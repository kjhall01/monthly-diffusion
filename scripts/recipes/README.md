# Historical experiment recipes

These shell launchers retain the existing experiment parameters and paired-AE
training. They find the repository root from their own location. Set `MD_PYTHON`
when a recipe needs a particular environment; inspect its source and data paths
before starting a run. Python targets now live in `research/` or `legacy/`.

For the separate atmosphere → ocean → diffusion sequence, use the
[main workflow](../README.md).
