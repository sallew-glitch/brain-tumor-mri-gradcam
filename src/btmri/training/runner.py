"""Run a whole experiment: one training run per test fold, then a cross-validation summary.

Output layout::

    outputs/runs/<name>_<timestamp>/
        config.yaml          resolved config (exact settings of this run)
        train.log
        summary.json         mean +/- std over folds, plus pooled out-of-fold metrics
        fold_1/
            best.pt          best checkpoint (by validation score)
            history.csv      per-epoch losses and scores
            test_metrics.json
            predictions.csv  per-image probabilities on the test fold
            curves.png, confusion_matrix.png
        fold_2/ ...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from btmri.config import ExperimentConfig, save_config
from btmri.constants import CLASS_NAMES, NUM_CLASSES
from btmri.data.dataset import build_dataloaders, load_split
from btmri.models import build_model
from btmri.training.metrics import SCALAR_METRICS, compute_metrics
from btmri.training.trainer import Trainer
from btmri.utils import get_device, set_seed
from btmri.utils.log import setup_logging
from btmri.viz import plot_confusion_matrix, plot_history

logger = logging.getLogger(__name__)


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def run_fold(cfg: ExperimentConfig, test_fold: int, fold_dir: Path) -> dict:
    fold_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed + test_fold)
    device = get_device(cfg.device)

    split = load_split(cfg, test_fold)
    logger.info(
        "Fold %d on %s | train %d imgs (%d patients) | val %d (fold %d) | test %d",
        test_fold, device, len(split.train), split.train["patient_id"].nunique(),
        len(split.val), split.val_fold, len(split.test),
    )
    loaders = build_dataloaders(cfg, split, device)

    model = build_model(cfg.model, NUM_CLASSES)
    class_weights = (
        inverse_frequency_weights(split.train["label"].to_numpy(), NUM_CLASSES)
        if cfg.train.class_weighted_loss
        else None
    )
    trainer = Trainer(
        model, cfg.train, device, fold_dir, CLASS_NAMES, class_weights,
        checkpoint_extra={"config": cfg.to_dict(), "test_fold": test_fold},
    )
    history = trainer.fit(loaders["train"], loaders["val"])

    preds = trainer.predict(loaders["test"])
    metrics = compute_metrics(preds["y_true"], preds["y_prob"], CLASS_NAMES)
    metrics["loss"] = preds["loss"]
    metrics["best_epoch"] = trainer.best_epoch

    (fold_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    predictions = pd.DataFrame(preds["y_prob"], columns=[f"prob_{c}" for c in CLASS_NAMES])
    predictions.insert(0, "image_id", preds["image_id"])
    predictions.insert(1, "label", preds["y_true"])
    predictions.insert(2, "pred", preds["y_prob"].argmax(1))
    predictions["test_fold"] = test_fold
    predictions.to_csv(fold_dir / "predictions.csv", index=False)
    plot_history(history, fold_dir / "curves.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, fold_dir / "confusion_matrix.png")

    logger.info(
        "Fold %d TEST | acc %.4f | balanced acc %.4f | macro-F1 %.4f | AUC %.4f",
        test_fold, metrics["accuracy"], metrics["balanced_accuracy"],
        metrics["macro_f1"], metrics["macro_auc"],
    )
    return metrics


def summarize_folds(fold_metrics: dict[int, dict], run_dir: Path) -> dict:
    """Mean/std across folds, plus metrics pooled over all out-of-fold predictions."""
    per_fold = pd.DataFrame(
        {fold: {k: m[k] for k in SCALAR_METRICS} for fold, m in fold_metrics.items()}
    ).T
    per_fold.index.name = "test_fold"
    per_fold.to_csv(run_dir / "per_fold_metrics.csv")

    pooled_preds = pd.concat(
        pd.read_csv(run_dir / f"fold_{fold}" / "predictions.csv") for fold in fold_metrics
    )
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    pooled = compute_metrics(
        pooled_preds["label"].to_numpy(), pooled_preds[prob_cols].to_numpy(), CLASS_NAMES
    )

    summary = {
        "folds": list(fold_metrics),
        "mean": per_fold.mean().to_dict(),
        "std": per_fold.std(ddof=0).to_dict(),
        "pooled_out_of_fold": pooled,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_experiment(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{cfg.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    setup_logging(run_dir / "train.log")
    logger.info("Run directory: %s", run_dir)

    fold_metrics = {fold: run_fold(cfg, fold, run_dir / f"fold_{fold}") for fold in cfg.folds}
    summary = summarize_folds(fold_metrics, run_dir)
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc"):
        logger.info("CV %-18s %.4f +/- %.4f", metric, summary["mean"][metric], summary["std"][metric])
    return run_dir
