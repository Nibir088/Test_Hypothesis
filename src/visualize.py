"""Visualize segmentation predictions from the trained model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from src.models import SimpleSegmentationNet
from src.oxford_pet_sam import load_oxford_pet_dataset


IMAGE_SIZE = (224, 224)


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model_transform() -> Compose:
    return Compose([
        lambda img: img.convert("RGB"),
        Resize(IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_binary_mask(mask: Image.Image) -> np.ndarray:
    mask_arr = np.array(mask)
    return (mask_arr == 1).astype(np.uint8)


def visualize_samples(
    root: str,
    model_checkpoint: str,
    device: str,
    num_samples: int = 3,
    start_index: int = 0,
    save_path: str = "visualization.png",
) -> None:
    root_path = Path(root).expanduser().resolve()
    dataset = load_oxford_pet_dataset(
        root=str(root_path),
        download=False,
        transform=lambda image: image.convert("RGB"),
        target_transform=lambda mask: Image.fromarray(np.array(mask)),
    )

    model = SimpleSegmentationNet(num_classes=1)
    checkpoint_path = Path(model_checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    transform = build_model_transform()
    figure, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row, idx in enumerate(range(start_index, start_index + num_samples)):
            if idx >= len(dataset):
                break
            image, mask = dataset[idx]
            input_tensor = transform(image).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prediction = (probs > 0.5).astype(np.uint8)
            gt_mask = make_binary_mask(mask)

            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f"Image #{idx}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(gt_mask, cmap="gray")
            axes[row, 1].set_title("Ground truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(prediction, cmap="gray")
            axes[row, 2].set_title("Predicted mask")
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    try:
        plt.show()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize segmentation outputs from the trained model.")
    parser.add_argument("--root", type=str, required=True, help="Project root containing the dataset directories.")
    parser.add_argument("--model-checkpoint", type=str, default="checkpoints/simple_segmentation.pth", help="Path to the trained segmentation model checkpoint.")
    parser.add_argument("--device", type=str, default=get_default_device(), help="Device for model inference.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to visualize.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for the dataset samples.")
    parser.add_argument("--save-path", type=str, default="visualization.png", help="Path to save the visualization figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_samples(
        root=args.root,
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        num_samples=args.num_samples,
        start_index=args.start_index,
        save_path=args.save_path or None,
    )


if __name__ == "__main__":
    main()
