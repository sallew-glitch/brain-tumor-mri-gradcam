"""Does the Grad-CAM heatmap point at the tumor? Scored against the expert tumor masks.

* ``pointing_hit``: is the single hottest heatmap pixel inside the tumor? (pointing game)
* ``energy_in_mask``: share of total heatmap mass that falls inside the tumor.
  Compare it with ``mask_area_frac``: a heatmap that ignores the tumor would score about
  the tumor's share of the image, so ``energy_in_mask / mask_area_frac`` > 1 means the
  model concentrates on the tumor more than chance.
* ``iou``: overlap between the thresholded heatmap and the mask.

High classification accuracy with low localization would suggest the model is using
shortcuts (e.g. skull shape, slice position) rather than the tumor itself.
"""

from __future__ import annotations

import numpy as np
import torch

LOCALIZATION_METRICS = ("pointing_hit", "energy_in_mask", "energy_ratio_vs_chance", "iou")


def localization_metrics(cam: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray,
                         threshold: float = 0.5) -> dict[str, float]:
    """``cam``: (H, W) in [0, 1]. ``mask``: (H, W) binary, same size."""
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask) > 0.5
    area = mask.mean()
    if area == 0:
        return {"mask_area_frac": 0.0, **{k: float("nan") for k in LOCALIZATION_METRICS}}

    total = cam.sum()
    energy = float(cam[mask].sum() / total) if total > 0 else 0.0
    hot = cam >= threshold
    union = (hot | mask).sum()
    return {
        "mask_area_frac": float(area),
        "pointing_hit": float(mask.flat[cam.argmax()]),
        "energy_in_mask": energy,
        "energy_ratio_vs_chance": energy / float(area),
        "iou": float((hot & mask).sum() / union) if union else 0.0,
    }
