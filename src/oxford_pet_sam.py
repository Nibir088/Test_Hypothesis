"""Dataset and SAM helper utilities for the Oxford-IIIT Pet experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple
import urllib.request

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
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> OxfordIIITPet:
    root_path = Path(root).expanduser().resolve()
    if split != "trainval":
        raise ValueError("Oxford-IIIT Pet only supports split='trainval'.")

    if transform is None:
        transform = lambda img: img.convert("RGB")
    if target_transform is None:
        target_transform = lambda mask: Image.fromarray(np.array(mask))

    return OxfordIIITPet(
        root=str(root_path),
        split=split,
        target_types=target_types,
        download=download,
        transform=transform,
        target_transform=target_transform,
    )


DEFAULT_SAM_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
DEFAULT_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def resolve_sam_checkpoint(
    checkpoint_path: Optional[str] = None,
    default_checkpoint_path: Optional[str] = None,
    download: bool = True,
) -> Path:
    explicit_candidate = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    default_candidate = Path(default_checkpoint_path or DEFAULT_SAM_CHECKPOINT).expanduser().resolve() if (default_checkpoint_path or DEFAULT_SAM_CHECKPOINT) else None

    for candidate in [explicit_candidate, default_candidate]:
        if candidate is not None and candidate.exists():
            return candidate

    download_target = default_candidate or explicit_candidate or Path(DEFAULT_SAM_CHECKPOINT).expanduser().resolve()
    if download:
        download_target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SAM checkpoint to {download_target}...")
        urllib.request.urlretrieve(DEFAULT_SAM_URL, str(download_target))
        if download_target.exists():
            return download_target

    raise FileNotFoundError(f"SAM checkpoint not found: {download_target}")


def build_sam_predictor(
    checkpoint_path: Optional[str] = None,
    model_type: str = "vit_b",
    device: Optional[str] = None,
    default_checkpoint_path: Optional[str] = None,
    download: bool = True,
) -> "SamPredictor":
    if SamPredictor is None or sam_model_registry is None:
        raise RuntimeError("segment-anything is not installed. Install it from PyPI.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = resolve_sam_checkpoint(
        checkpoint_path=checkpoint_path,
        default_checkpoint_path=default_checkpoint_path,
        download=download,
    )

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
