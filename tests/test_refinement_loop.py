import numpy as np

from src.experiments import compute_dice, run_closed_loop_refinement


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
