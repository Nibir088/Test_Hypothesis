# Test Hypothesis

This repository contains an Oxford-IIIT Pet + Segment Anything Model (SAM) experiment scaffold.

## Goal

- Keep dataset and checkpoint files separate from repository history
- Provide a runnable codebase for point-based segmentation prompting and refinement
- Support a minimal command-line entry point for sanity checks and later experiment stages

## Repository layout

- `src/` — Python package with dataset loading, SAM helpers, and model scaffolds
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes data directories, virtual environments, and generated files

## Data separation

This repository intentionally keeps training data and checkpoints outside version control.

Ignored directories:

- `oxford-iiit-pet/`
- `checkpoints/`
- `.venv/`
- `__pycache__/`

## Quick start

1. Create or activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place data and checkpoints in their directories (not committed):

- `oxford-iiit-pet/`
- `checkpoints/`

### Download the Oxford-IIIT Pet dataset

You can let the script download the dataset for you, or download it manually and place it into `oxford-iiit-pet/`.

To download automatically via the script:

```bash
python -m src.experiments --stage sanity --root . --download
```

This will download the dataset into the local project root under `oxford-iiit-pet/`.

If you want to download manually, place the extracted dataset files into `oxford-iiit-pet/`.

### Download the SAM checkpoint

This repository does not include the `.pth` checkpoint file. You should download a SAM checkpoint such as `sam_vit_b_01ec64.pth` from the official source you have access to, then save it into the `checkpoints/` directory.

Example:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

If you prefer `wget`:

```bash
wget -O checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

4. Run a sanity stage:

```bash
python -m src.experiments --stage sanity --root . --checkpoint checkpoints/sam_vit_b_01ec64.pth
```

### Train the segmentation model

```bash
python -m src.experiments --stage train --root . --download
```

### Evaluate the segmentation model

```bash
python -m src.experiments --stage eval --root . --model-checkpoint checkpoints/simple_segmentation.pth
```

### Visualize predictions

```bash
python -m src.visualize --root . --model-checkpoint checkpoints/simple_segmentation.pth --num-samples 3
```

### Visualize refinement steps

Train the model while saving per-step refinement snapshots:

```bash
python -m src.experiments --stage train --root . --download --model-checkpoint checkpoints/simple_segmentation.pth --progress-dir training_progress
```

This will save one composite image per sample in the `training_progress/` directory, showing the original image, the mask, and the intermediate refinement outputs for each step (for example, `sample_0000_progress.png`).

### Run the SAM-style point refinement loop

The refinement stage now uses a point prompt and passes it directly to a SAM-style predictor to generate a mask, then scores that mask against the target and iteratively updates the point for $T = 1, 2, 3, 5$.

```bash
python -m src.experiments --stage refine --root . --model-checkpoint checkpoints/simple_segmentation.pth --sample-index 0 --refine-output-dir refinement_outputs
```

This writes visualization files such as:

- `refinement_outputs/refinement_T1.png`
- `refinement_outputs/refinement_T2.png`
- `refinement_outputs/refinement_T3.png`
- `refinement_outputs/refinement_T5.png`

## Notes

- The `src` package is import-safe and focuses on a clean experiment scaffold.
- Use `download=True` with `load_oxford_pet_dataset` only if you want the dataset to be downloaded locally.
