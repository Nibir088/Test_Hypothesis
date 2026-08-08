"""Experiment routines for the Test Hypothesis segmentation scaffold."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from src.models import SimpleSegmentationNet
from src.oxford_pet_sam import (
    build_sam_predictor,
    get_mask_from_target,
    load_oxford_pet_dataset,
    prepare_sam_image,
)


IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
MODEL_CHECKPOINT = "checkpoints/simple_segmentation.pth"


def build_image_transform() -> Compose:
    return Compose([
        lambda img: img.convert("RGB"),
        Resize(IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_mask_transform() -> Compose:
    return Compose([
        lambda mask: Image.fromarray((np.array(mask) == 1).astype('uint8') * 255),
        Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
        ToTensor(),
    ])


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


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_data_loaders(root: str, download: bool, device: str, batch_size: int = BATCH_SIZE, num_workers: int = 4):
    image_transform = build_image_transform()
    mask_transform = build_mask_transform()
    dataset = load_oxford_pet_dataset(
        root=root,
        download=download,
        transform=image_transform,
        target_transform=mask_transform,
    )
    total = len(dataset)
    val_size = int(total * 0.2)
    train_size = total - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_epoch(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    batch_count = len(loader)
    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)
        loss = criterion(predictions, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

        if batch_idx % 10 == 0 or batch_idx == batch_count:
            print(f"    batch {batch_idx}/{batch_count} - loss={loss.item():.4f}")
    return total_loss / len(loader.dataset)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = model(images)
            loss = criterion(predictions, masks)
            total_loss += loss.item() * images.size(0)
            predicted_mask = (predictions > 0.0).float()
            intersection = (predicted_mask * masks).sum(dim=(1, 2, 3))
            union = ((predicted_mask + masks) > 0).float().sum(dim=(1, 2, 3))
            total_iou += (intersection / union.clamp(min=1e-6)).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    avg_iou = total_iou / len(loader.dataset)
    return avg_loss, avg_iou


def run_train(root: str, download: bool, checkpoint: str, model_checkpoint: str, device: str) -> None:
    train_loader, val_loader = get_data_loaders(root=root, download=download, device=device)
    model = build_model(device)
    pos_weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on device={device} with {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} val samples.")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

    checkpoint_path = Path(model_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved segmentation model weights to: {checkpoint_path}")

    if checkpoint:
        print("Running sanity check with SAM using provided checkpoint after training.")
        run_sanity_check(root, checkpoint, download=False)


def run_eval(root: str, download: bool, model_checkpoint: str, device: str) -> None:
    model = build_model(device)
    checkpoint_path = Path(model_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model weights from: {checkpoint_path}")

    _, val_loader = get_data_loaders(root=root, download=download, device=device)
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_iou = evaluate_model(model, val_loader, criterion, device)
    print(f"Evaluation result - val_loss={val_loss:.4f} val_iou={val_iou:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Oxford Pet + SAM experiment scaffold.")
    parser.add_argument("--stage", choices=["sanity", "train", "eval"], required=True)
    parser.add_argument("--root", type=str, required=True, help="Project root containing the dataset directories.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a SAM checkpoint for sanity or inference.")
    parser.add_argument("--model-checkpoint", type=str, default=MODEL_CHECKPOINT, help="Path to save or load the segmentation model weights.")
    parser.add_argument("--download", action="store_true", help="Download the Oxford-IIIT Pet dataset if missing.")
    parser.add_argument("--device", type=str, default=get_default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = Path(args.root).expanduser().resolve()

    if args.stage == "sanity":
        run_sanity_check(str(root_path), args.checkpoint or None, args.download)
        return

    if args.stage == "train":
        run_train(str(root_path), args.download, args.checkpoint, args.model_checkpoint, args.device)
    elif args.stage == "eval":
        run_eval(str(root_path), args.download, args.model_checkpoint, args.device)


if __name__ == "__main__":
    main()
