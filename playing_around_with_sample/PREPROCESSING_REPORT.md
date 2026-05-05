# Preprocessing Pipeline Summary

## Goal
Build a reproducible image preprocessing pipeline for geolocation model training where all model inputs are fixed to 224x224, while preserving geographic labels and increasing training diversity through random spatial sampling.

## Data Source And Initial Export
- Source dataset: `hf://datasets/gydou/released_img/data/train-00000-of-00001.parquet`
- Export script: `collect-sample-data.py`
- Output folder: `sample_data/`
- Generated assets:
  - `sample_data/images/*.jpg`
  - `sample_data/metadata.csv` with `image_file`, `Latitude`, `Longitude`

## Preprocessing Script
- Main script: `preprocessing.py`
- Input defaults:
  - `--input-root sample_data`
- Output defaults:
  - `--output-root preprocessed`

## Pipeline Evolution
1. Initial version
- Generated 3 random 224x224 crops per image.

2. Context-preserving crop strategy
- Updated to first crop a square region and then resize to 224x224.
- Square side initially set to the shortest image side.

3. Orientation fix
- Added EXIF orientation normalization with `ImageOps.exif_transpose` to avoid 90-degree rotated outputs.

4. Final crop variability update (current)
- Number of variants increased from 3 to 5 per source image.
- For each variant, square side length is randomly sampled from:
  - `50%` to `100%` of the shortest side of the oriented image.
- That random square is cropped at a random valid location.
- The square crop is resized to `224x224` using LANCZOS resampling.

## Current Output Structure
- Images: `preprocessed/images/`
- Metadata: `preprocessed/metadata.csv`

Naming pattern for generated images:
- Source: `img_00001.jpg`
- Variants: `img_00001_0.jpg` through `img_00001_4.jpg`

## Metadata Schema (Current)
The preprocessing metadata keeps labels and now records crop provenance:
- `image_file`: relative path to generated image (for example `images/img_00001_2.jpg`)
- `source_image_file`: relative path to source image in input root (for example `images/img_00001.jpg`)
- `variant_index`: integer variant id (`0-4` by default)
- `Latitude`: inherited from source metadata
- `Longitude`: inherited from source metadata
- `orig_width`: oriented source image width used for crop sampling
- `orig_height`: oriented source image height used for crop sampling
- `crop_size`: sampled square side length before resize
- `crop_left`: left pixel coordinate of square crop in oriented source image
- `crop_top`: top pixel coordinate of square crop in oriented source image
- `resize_width`: output width (always `224`)
- `resize_height`: output height (always `224`)

## Reproducibility
- The script supports `--seed` to make crop sampling deterministic.
- Without a seed, random crop choices vary between runs.

## Notes For Final Project Report
- This preprocessing approach performs data augmentation by exposing each scene under multiple spatial windows and scales.
- Orientation normalization ensures geometric consistency across all training examples.
- Provenance metadata enables auditability, debugging, and optional ablation studies on crop scale and position.
