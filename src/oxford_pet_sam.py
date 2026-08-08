"""Dataset and SAM helper utilities for the Oxford-IIIT Pet experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:  # pragma: no cover
    SamPredictor = None
    sam_model_registry = None


def load_oxford_pet_dataset(
    root: str,
    split: str = "trainval",
    download: bool = False,
    target_types: Tuple[str, ...] = ("segmentation",),
) -> OxfordIIITPet:
    root_path = Path(root).expanduser().resolve()
    if split != "trainval":
        raise ValueError("Oxford-IIIT Pet only supports split='trainval'.")

    return OxfordIIITPet(
        root=str(root_path),
        split=split,
        target_types=target_types,
        download=download,
        transform=lambda img: img.convert("RGB"),
        target_transform=lambda mask: Image.fromarray(np.array(mask)),
    )


def build_sam_predictor(
    checkpoint_path: str,
    model_type: str = "vit_b",
    device: Optional[str] = None,
) -> "SamPredictor":
    if SamPredictor is None or sam_model_registry is None:
        raise RuntimeError("segment-anything is not installed. Install it from PyPI.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device)
    predictor = SamPredictor(sam)
    return predictor


def prepare_sam_image(image: Image.Image) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL.Image for SAM input.")
    return np.asarray(image)


def get_mask_from_target(target) -> np.ndarray:
    if isinstance(target, Image.Image):
        return np.asarray(target.convert("L"))
    if isinstance(target, np.ndarray):
        return target
    raise TypeError(f"Unsupported target type: {type(target)}")
