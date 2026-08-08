"""Experiment routines for the Test Hypothesis segmentation scaffold."""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
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
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
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
            probs = torch.sigmoid(predictions)
            predicted_mask = (probs > 0.5).float()
            intersection = (predicted_mask * masks).sum(dim=(1, 2, 3))
            union = ((predicted_mask + masks) > 0).float().sum(dim=(1, 2, 3))
            total_iou += (intersection / union.clamp(min=1e-6)).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    avg_iou = total_iou / len(loader.dataset)
    return avg_loss, avg_iou


def compute_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    if pred_mask.shape != target_mask.shape:
        target_mask = np.array(Image.fromarray(target_mask.astype("uint8")).resize(pred_mask.shape[::-1], Image.NEAREST)) > 0
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def run_closed_loop_refinement(
    predicted_masks: list[np.ndarray],
    target_mask: np.ndarray,
    initial_point: tuple[int, int],
) -> dict[str, object]:
    point = initial_point
    history: list[dict[str, object]] = []
    for step_idx, mask in enumerate(predicted_masks, start=1):
        dice = compute_dice(mask, target_mask)
        reward = dice - (0.01 * (abs(point[0]) + abs(point[1])))
        history.append({"step": step_idx, "dice": dice, "reward": reward, "point": point})
        if step_idx < len(predicted_masks):
            point = (point[0] + 1, point[1] + 1)
    return {"steps": history, "final_point": point}


def build_point_prior(mask_shape: Tuple[int, int], point: tuple[int, int], sigma: float = 0.08) -> np.ndarray:
    height, width = mask_shape
    y_coords = np.linspace(0, height - 1, height)
    x_coords = np.linspace(0, width - 1, width)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    distance = (xx - point[0]) ** 2 + (yy - point[1]) ** 2
    scale = max(height, width) * sigma
    prior = np.exp(-distance / (2.0 * scale * scale))
    return prior / prior.max()


def generate_prompted_mask(
    model: torch.nn.Module,
    image: Image.Image,
    point: tuple[int, int],
    device: str,
) -> np.ndarray:
    transform = build_image_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prior = build_point_prior(probs.shape, point)
    combined = probs * prior
    return (combined > 0.5).astype(np.uint8)


def run_prompt_refinement_loop(
    model: torch.nn.Module,
    image: Image.Image,
    target_mask: np.ndarray,
    initial_point: tuple[int, int],
    max_steps: int,
    device: str,
) -> dict[str, object]:
    point = initial_point
    history: list[dict[str, object]] = []
    prev_dice: Optional[float] = None

    for step_idx in range(1, max_steps + 1):
        prompted_mask = generate_prompted_mask(model, image, point, device)
        dice = compute_dice(prompted_mask, target_mask)
        reward = 0.0 if prev_dice is None else dice - prev_dice
        history.append({
            "step": step_idx,
            "point": point,
            "mask": prompted_mask,
            "dice": dice,
            "reward": reward,
        })

        if step_idx < max_steps:
            if prev_dice is None or dice >= prev_dice:
                centroid = np.argwhere(prompted_mask > 0)
                if centroid.size == 0:
                    point = (point[0] + 3, point[1] + 3)
                else:
                    ys, xs = centroid[:, 0], centroid[:, 1]
                    point = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
            else:
                point = (point[0] + 5, point[1] + 5)
            prev_dice = dice

    return {"steps": history, "final_point": point}


def save_refinement_loop_plot(
    image: Image.Image,
    target_mask: np.ndarray,
    refinement_result: dict[str, object],
    sample_idx: int,
    save_path: Path,
) -> None:
    steps = refinement_result["steps"]
    fig, axes = plt.subplots(len(steps), 3, figsize=(12, 3 * len(steps)))
    if len(steps) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, entry in enumerate(steps):
        point = entry["point"]
        mask = entry["mask"]
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Step {entry['step']} - Image")
        axes[row, 0].axis("off")
        axes[row, 0].scatter(point[0], point[1], color="red", s=60, marker="x", linewidths=2)
        axes[row, 1].imshow(target_mask, cmap="gray")
        axes[row, 1].set_title("Mask")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(mask, cmap="gray")
        axes[row, 2].set_title(f"$\\bar{{M}}_{{{entry['step']}}}$")
        axes[row, 2].axis("off")

    fig.suptitle(f"Refinement loop for sample {sample_idx}", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_refinement_inference(
    root: str,
    model_checkpoint: str,
    device: str,
    sample_idx: int = 0,
    output_dir: str = "refinement_outputs",
) -> None:
    model = build_model(device)
    checkpoint_path = Path(model_checkpoint).expanduser().resolve()
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Model checkpoint not found at {checkpoint_path}; using an initialized model for refinement inference.")
    model.eval()

    dataset = load_oxford_pet_dataset(root=root, download=False)
    image, mask = dataset[sample_idx]
    target_mask = (np.array(mask) == 1).astype(np.uint8)
    if target_mask.shape != (IMAGE_SIZE[0], IMAGE_SIZE[1]):
        target_mask = np.array(Image.fromarray(target_mask).resize(IMAGE_SIZE, Image.NEAREST)) > 0
    initial_point = get_selected_point(target_mask)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running refinement loop for sample {sample_idx} with initial point {initial_point}")
    for max_steps in [1, 2, 3, 5]:
        result = run_prompt_refinement_loop(
            model=model,
            image=image,
            target_mask=target_mask,
            initial_point=initial_point,
            max_steps=max_steps,
            device=device,
        )
        final_dice = result["steps"][-1]["dice"]
        print(f"T={max_steps} -> final_dice={final_dice:.4f} final_point={result['final_point']}")
        save_refinement_loop_plot(
            image=image,
            target_mask=target_mask,
            refinement_result=result,
            sample_idx=sample_idx,
            save_path=output_path / f"refinement_T{max_steps}.png",
        )
        print(f"Saved refinement visualization to {output_path / f'refinement_T{max_steps}.png'}")


def get_selected_point(mask: np.ndarray) -> tuple[int, int]:
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        h, w = mask.shape
        return w // 2, h // 2
    ys, xs = foreground[:, 0], foreground[:, 1]
    center_y = int(round(float(ys.mean())))
    center_x = int(round(float(xs.mean())))
    return center_x, center_y


def save_training_progress_samples(
    model: torch.nn.Module,
    root: str,
    epoch: int,
    sample_indices: list[int],
    device: str,
    output_dir: str,
) -> None:
    model.eval()
    dataset = load_oxford_pet_dataset(root=root, download=False)
    save_dir = Path(output_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "progress_history.pkl"
    transform = build_image_transform()

    history: dict[str, list[dict[str, object]]] = {}
    if history_path.exists():
        with history_path.open("rb") as handle:
            history = pickle.load(handle)

    with torch.no_grad():
        for idx in sample_indices:
            if idx >= len(dataset):
                break
            image, mask = dataset[idx]
            input_tensor = transform(image).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            predicted_mask = (probs > 0.5).astype(np.uint8)
            gt_mask = (np.array(mask) == 1).astype(np.uint8)
            selected_point = get_selected_point(gt_mask)

            sample_history = history.setdefault(str(idx), [])
            sample_history.append({"epoch": epoch, "predicted_mask": predicted_mask})

            row_count = max(1, len(sample_history))
            col_count = 4
            fig, axes = plt.subplots(row_count, col_count, figsize=(16, 4 * row_count))
            if row_count == 1:
                axes = np.expand_dims(axes, axis=0)

            for row, entry in enumerate(sample_history):
                row_epoch = int(entry["epoch"])
                row_pred = entry["predicted_mask"]
                axes[row, 0].imshow(image)
                axes[row, 0].set_title(f"Image #{idx}")
                axes[row, 0].axis("off")
                axes[row, 0].scatter(selected_point[0], selected_point[1], color="red", s=70, marker="x", linewidths=2)
                axes[row, 0].text(
                    selected_point[0] + 3,
                    selected_point[1] - 3,
                    f"({selected_point[0]},{selected_point[1]})",
                    color="white",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "red", "alpha": 0.7},
                )
                axes[row, 1].imshow(gt_mask, cmap="gray")
                axes[row, 1].set_title("Mask")
                axes[row, 1].axis("off")

                for col_idx in range(2, col_count):
                    panel_idx = col_idx - 1
                    panel_mask = row_pred if panel_idx == 1 else row_pred
                    axes[row, col_idx].imshow(panel_mask, cmap="gray")
                    axes[row, col_idx].set_title(f"$\\bar{{M}}_{{{panel_idx}}}$")
                    axes[row, col_idx].axis("off")

                fig.suptitle(f"Refinement steps for sample {idx} (epoch {row_epoch})", fontsize=12)

            plt.tight_layout()
            sample_path = save_dir / f"sample_{idx:04d}_progress.png"
            fig.savefig(sample_path, dpi=150)
            plt.close(fig)
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Updated single-image progress view for sample {idx} at {sample_path}")

    with history_path.open("wb") as handle:
        pickle.dump(history, handle)


def run_train(root: str, download: bool, checkpoint: str, model_checkpoint: str, device: str, progress_dir: str = "training_progress") -> None:
    train_loader, val_loader = get_data_loaders(root=root, download=download, device=device)
    model = build_model(device)
    pos_weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on device={device} with {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} val samples.")

    history_path = Path(progress_dir).expanduser().resolve() / "progress_history.pkl"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = evaluate_model(model, val_loader, criterion, device)
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Epoch {epoch}/{NUM_EPOCHS} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}")
        save_training_progress_samples(
            model=model,
            root=root,
            epoch=epoch,
            sample_indices=[0, 1, 2],
            device=device,
            output_dir=progress_dir,
        )

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
    parser.add_argument("--stage", choices=["sanity", "train", "eval", "refine"], required=True)
    parser.add_argument("--root", type=str, required=True, help="Project root containing the dataset directories.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a SAM checkpoint for sanity or inference.")
    parser.add_argument("--model-checkpoint", type=str, default=MODEL_CHECKPOINT, help="Path to save or load the segmentation model weights.")
    parser.add_argument("--download", action="store_true", help="Download the Oxford-IIIT Pet dataset if missing.")
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--progress-dir", type=str, default="training_progress", help="Directory to save epoch-wise sample prediction snapshots during training.")
    parser.add_argument("--refine-output-dir", type=str, default="refinement_outputs", help="Directory to save per-step refinement visualizations for T=1,2,3,5.")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index to use for the refinement inference loop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = Path(args.root).expanduser().resolve()

    if args.stage == "sanity":
        run_sanity_check(str(root_path), args.checkpoint or None, args.download)
        return

    if args.stage == "train":
        run_train(
            str(root_path),
            args.download,
            args.checkpoint,
            args.model_checkpoint,
            args.device,
            progress_dir=args.progress_dir,
        )
    elif args.stage == "eval":
        run_eval(str(root_path), args.download, args.model_checkpoint, args.device)
    elif args.stage == "refine":
        run_refinement_inference(
            root=str(root_path),
            model_checkpoint=args.model_checkpoint,
            device=args.device,
            sample_idx=args.sample_index,
            output_dir=args.refine_output_dir,
        )


if __name__ == "__main__":
    main()
