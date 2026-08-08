"""Experiment routines for the Test Hypothesis segmentation scaffold."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.models import SimpleSegmentationNet
from src.oxford_pet_sam import (
    build_sam_predictor,
    get_mask_from_target,
    load_oxford_pet_dataset,
    prepare_sam_image,
)


def run_sanity_check(root: str, checkpoint: Optional[str], download: bool) -> None:
    dataset = load_oxford_pet_dataset(root=root, download=download)
    print(f"Loaded Oxford-IIIT Pet dataset with {len(dataset)} samples.")

    sample_image, sample_mask = dataset[0]
    print(f"Sample image type: {type(sample_image)}, size={sample_image.size}")
    mask_array = get_mask_from_target(sample_mask)
    print(f"Sample mask shape: {mask_array.shape}, dtype={mask_array.dtype}")

    if checkpoint:
        predictor = build_sam_predictor(checkpoint)
        image_np = prepare_sam_image(sample_image)
        predictor.set_image(image_np)
        print("SAM predictor created and image prepared successfully.")
    else:
        print("No SAM checkpoint provided; skipping SAM predictor initialization.")


def build_model(device: str) -> torch.nn.Module:
    model = SimpleSegmentationNet(num_classes=1)
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Oxford Pet + SAM experiment scaffold.")
    parser.add_argument("--stage", choices=["sanity", "train", "eval"], required=True)
    parser.add_argument("--root", type=str, required=True, help="Project root containing the dataset directories.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a SAM checkpoint for sanity or inference.")
    parser.add_argument("--download", action="store_true", help="Download the Oxford-IIIT Pet dataset if missing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = Path(args.root).expanduser().resolve()

    if args.stage == "sanity":
        run_sanity_check(str(root_path), args.checkpoint or None, args.download)
        return

    model = build_model(args.device)
    print(f"Built model on device: {args.device}")

    if args.stage == "train":
        print("Train stage is not implemented yet. Use sanity as a starter.")
    elif args.stage == "eval":
        print("Eval stage is not implemented yet. Use sanity as a starter.")


if __name__ == "__main__":
    main()
