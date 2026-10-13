"""Run Grad-CAM over one fold of a trained run and score it against the tumor masks.

Writes to ``<run_dir>/fold_<k>/gradcam_<split>_<layer>/``:
    localization.csv   one row per image (prediction + localization metrics)
    summary.json       averages overall, per class, and for correct vs. wrong predictions
    examples.png       heatmaps for a sample of correct and misclassified images

Use ``split="val"`` when comparing models or layers, and ``split="test"`` only for the final
report, so the test patients never influence which model you pick.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
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

SPLITS = ("test", "val")


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    """Rebuild the model from the config stored inside the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: ExperimentConfig = config_from_dict(checkpoint["config"])
    cfg.model.pretrained = False  # weights come from the checkpoint, no download needed
    model = build_model(cfg.model, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), cfg, checkpoint


def gradcam_dir_name(split: str, layer: str | None) -> str:
    return f"gradcam_{split}_{(layer or 'last').replace('.', '-')}"


def _pick_examples(results: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Up to half misclassified images, the rest correct ones, chosen at random."""
    wrong = results[~results["correct"]]
    right = results[results["correct"]]
    n_wrong = min(len(wrong), n // 2)
    chosen = pd.concat([
        right.sample(min(len(right), n - n_wrong), random_state=seed),
        wrong.sample(n_wrong, random_state=seed),
    ])
    return chosen["image_id"].astype(int).tolist()


def _plot_examples(gradcam: GradCAM, dataset: BrainTumorDataset, image_ids: list[int],
                   class_names: list[str], target: str, device: torch.device, path: Path) -> None:
    index_of = {int(i): pos for pos, i in enumerate(dataset.metadata["image_id"])}
    batch = next(iter(DataLoader(Subset(dataset, [index_of[i] for i in image_ids]),
                                 batch_size=len(image_ids))))
    images, labels = batch["image"].to(device), batch["label"].to(device)
    cams, logits = gradcam(images, labels if target == "true" else None)
    preds = logits.argmax(1)
    titles = [
        f"#{image_id} true={class_names[int(labels[i])]}\npred={class_names[int(preds[i])]}"
        + ("" if labels[i] == preds[i] else "  (WRONG)")
        for i, image_id in enumerate(batch["image_id"].tolist())
    ]
    plot_image_grid(
        list(denormalize(images)[:, 0].cpu().numpy()), titles, path,
        masks=list(batch["mask"].numpy()), heatmaps=list(cams.cpu().numpy()),
    )


def explain_fold(
    run_dir: str | Path,
    fold: int,
    num_examples: int = 16,
    device: str = "auto",
    processed_dir: str | None = None,
    target: str = "predicted",
    split: str = "test",
    layer: str | None = None,
) -> dict:
    """``target``: explain the "predicted" class (what the model did) or the "true" class.
    ``split``: "test" or "val". ``layer``: see ``btmri.explain.gradcam``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    fold_dir = Path(run_dir) / f"fold_{fold}"
    out_dir = fold_dir / gradcam_dir_name(split, layer)
    out_dir.mkdir(exist_ok=True)
    dev = get_device(device)

    model, cfg, checkpoint = load_trained_model(fold_dir / "best.pt", dev)
    class_names = checkpoint["class_names"]
    if processed_dir:  # e.g. a model trained on Colab, explained on a laptop
        cfg.data.processed_dir = processed_dir

    metadata = getattr(load_split(cfg, fold), split)
    size = cfg.data.image_size
    dataset = BrainTumorDataset(
        metadata, cfg.data.processed_dir, build_eval_transform(size), return_mask=True, mask_size=size
    )
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False)

    gradcam = GradCAM(model, layer)
    patient_of = dict(zip(metadata["image_id"], metadata["patient_id"], strict=True))
    rows = []
    for batch in tqdm(loader, desc=f"Grad-CAM fold {fold} ({split}, {layer or 'last'})"):
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

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "localization.csv", index=False)
    metric_cols = ["mask_area_frac", *LOCALIZATION_METRICS]
    summary = {
        "split": split,
        "layer": gradcam.layer_name or "last",
        "target": target,
        "overall": results[metric_cols].mean().to_dict(),
        "by_true_class": results.groupby("label")[metric_cols].mean().to_dict(orient="index"),
        "by_correctness": results.groupby("correct")[metric_cols].mean()
                                 .rename(index=str).to_dict(orient="index"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if num_examples > 0:
        example_ids = _pick_examples(results, num_examples, cfg.seed)
        _plot_examples(gradcam, dataset, example_ids, class_names, target, dev, out_dir / "examples.png")

    overall = summary["overall"]
    logger.info(
        "Fold %d Grad-CAM (%s, %s) | pointing game %.3f | energy in tumor %.3f (%.1fx chance) | IoU %.3f",
        fold, split, summary["layer"], overall["pointing_hit"], overall["energy_in_mask"],
        overall["energy_ratio_vs_chance"], overall["iou"],
    )
    return summary
