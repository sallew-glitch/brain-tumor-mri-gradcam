"""Run Grad-CAM over a trained fold's test set and score it against the tumor masks.

Writes to ``<run_dir>/fold_<k>/gradcam/``:
    localization.csv   one row per test image (prediction + localization metrics)
    summary.json       averages overall, per class, and for correct vs. wrong predictions
    examples.png       heatmaps for a sample of correct and misclassified images
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from btmri.config import ExperimentConfig, config_from_dict
from btmri.data.dataset import BrainTumorDataset, load_split
from btmri.data.transforms import build_eval_transform, denormalize
from btmri.explain.gradcam import GradCAM
from btmri.explain.localization import LOCALIZATION_METRICS, localization_metrics
from btmri.models import build_model
from btmri.utils import get_device
from btmri.viz import plot_image_grid

logger = logging.getLogger(__name__)


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    """Rebuild the model from the config stored inside the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: ExperimentConfig = config_from_dict(checkpoint["config"])
    cfg.model.pretrained = False  # weights come from the checkpoint, no download needed
    model = build_model(cfg.model, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), cfg, checkpoint


def _pick_examples(predictions: pd.DataFrame, n: int, seed: int) -> set[int]:
    """Up to half misclassified images, the rest correct ones, chosen at random."""
    wrong = predictions[predictions["label"] != predictions["pred"]]
    right = predictions[predictions["label"] == predictions["pred"]]
    n_wrong = min(len(wrong), n // 2)
    chosen = pd.concat([
        wrong.sample(n_wrong, random_state=seed),
        right.sample(min(len(right), n - n_wrong), random_state=seed),
    ])
    return set(chosen["image_id"].astype(int))


def explain_fold(
    run_dir: str | Path,
    fold: int,
    num_examples: int = 16,
    device: str = "auto",
    processed_dir: str | None = None,
    target: str = "predicted",
) -> dict:
    """``target``: explain the "predicted" class (what the model did) or the "true" class."""
    fold_dir = Path(run_dir) / f"fold_{fold}"
    out_dir = fold_dir / "gradcam"
    out_dir.mkdir(exist_ok=True)
    dev = get_device(device)

    model, cfg, checkpoint = load_trained_model(fold_dir / "best.pt", dev)
    class_names = checkpoint["class_names"]
    if processed_dir:  # e.g. a model trained on Colab, explained on a laptop
        cfg.data.processed_dir = processed_dir

    test = load_split(cfg, fold).test
    size = cfg.data.image_size
    dataset = BrainTumorDataset(
        test, cfg.data.processed_dir, build_eval_transform(size), return_mask=True, mask_size=size
    )
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False)
    example_ids = _pick_examples(pd.read_csv(fold_dir / "predictions.csv"), num_examples, cfg.seed)

    gradcam = GradCAM(model)
    patient_of = dict(zip(test["image_id"], test["patient_id"], strict=True))
    rows, examples = [], []
    for batch in tqdm(loader, desc=f"Grad-CAM fold {fold}"):
        images, labels = batch["image"].to(dev), batch["label"].to(dev)
        cams, logits = gradcam(images, labels if target == "true" else None)
        preds = logits.argmax(1)
        for i, image_id in enumerate(batch["image_id"].tolist()):
            label, pred = int(labels[i]), int(preds[i])
            rows.append({
                "image_id": image_id,
                "patient_id": patient_of[image_id],
                "label": class_names[label],
                "pred": class_names[pred],
                "correct": label == pred,
                **localization_metrics(cams[i].cpu(), batch["mask"][i]),
            })
            if image_id in example_ids:
                examples.append({
                    "image": denormalize(images[i : i + 1])[0, 0].cpu().numpy(),
                    "cam": cams[i].cpu().numpy(),
                    "mask": batch["mask"][i].numpy(),
                    "title": f"#{image_id} true={class_names[label]}\npred={class_names[pred]}"
                             + ("" if label == pred else "  (WRONG)"),
                })

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "localization.csv", index=False)
    metric_cols = ["mask_area_frac", *LOCALIZATION_METRICS]
    summary = {
        "target": target,
        "overall": results[metric_cols].mean().to_dict(),
        "by_true_class": results.groupby("label")[metric_cols].mean().to_dict(orient="index"),
        "by_correctness": results.groupby("correct")[metric_cols].mean()
                                 .rename(index=str).to_dict(orient="index"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if examples:
        examples.sort(key=lambda e: "WRONG" in e["title"])
        plot_image_grid(
            [e["image"] for e in examples], [e["title"] for e in examples], out_dir / "examples.png",
            masks=[e["mask"] for e in examples], heatmaps=[e["cam"] for e in examples],
        )

    overall = summary["overall"]
    logger.info(
        "Fold %d Grad-CAM | pointing game %.3f | energy in tumor %.3f (%.1fx chance) | IoU %.3f",
        fold, overall["pointing_hit"], overall["energy_in_mask"],
        overall["energy_ratio_vs_chance"], overall["iou"],
    )
    return summary
