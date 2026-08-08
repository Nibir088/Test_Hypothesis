from pathlib import Path

import numpy as np

from src.experiments import compute_dice, run_closed_loop_refinement
from src.oxford_pet_sam import resolve_sam_checkpoint


def test_closed_loop_refinement_records_dice_and_rewards():
    gt_mask = np.zeros((8, 8), dtype=np.uint8)
    gt_mask[2:6, 2:6] = 1
    preds = [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8), dtype=np.uint8),
        gt_mask,
    ]

    result = run_closed_loop_refinement(preds, gt_mask, initial_point=(0, 0))

    assert len(result["steps"]) == 3
    assert result["steps"][0]["dice"] >= 0.0
    assert result["steps"][2]["dice"] == 1.0
    assert result["steps"][2]["reward"] >= 0.0
    assert result["final_point"] != (0, 0)
    assert compute_dice(gt_mask, gt_mask) == 1.0


def test_resolve_sam_checkpoint_falls_back_to_local_default(tmp_path: Path) -> None:
    explicit_checkpoint = tmp_path / "missing.pth"
    default_checkpoint = tmp_path / "default_sam.pth"
    default_checkpoint.write_bytes(b"weights")

    resolved = resolve_sam_checkpoint(
        str(explicit_checkpoint),
        default_checkpoint_path=str(default_checkpoint),
        download=False,
    )

    assert resolved == default_checkpoint
