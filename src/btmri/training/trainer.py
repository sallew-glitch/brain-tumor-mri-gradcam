"""The training loop.

One epoch = one pass over the training images. In each step the model predicts a batch,
the loss measures how wrong it was, backpropagation computes how each weight should change,
and the optimizer applies that change. After every epoch we score the validation fold, which
the model never trains on, and keep the checkpoint that scores best there.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from btmri.config import TrainConfig
from btmri.models.factory import parameter_groups, set_backbone_trainable
from btmri.training.metrics import compute_metrics

logger = logging.getLogger(__name__)

BEST_CHECKPOINT = "best.pt"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        device: torch.device,
        out_dir: str | Path,
        class_names: tuple[str, ...],
        class_weights: torch.Tensor | None = None,
        checkpoint_extra: dict[str, Any] | None = None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(out_dir)
        self.class_names = class_names
        self.checkpoint_extra = checkpoint_extra or {}

        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            parameter_groups(model, cfg.lr, cfg.backbone_lr_mult), weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_epoch: int | None = None

    # ------------------------------------------------------------------ training
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        higher_is_better = self.cfg.monitor != "loss"
        best_score = -math.inf if higher_is_better else math.inf
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            backbone_trainable = epoch > self.cfg.freeze_backbone_epochs
            set_backbone_trainable(self.model, backbone_trainable)

            train_loss, train_acc = self._train_one_epoch(train_loader, epoch)
            val = self.evaluate(val_loader)
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "backbone_trainable": backbone_trainable,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                **{f"val_{k}": v for k, v in val.items() if isinstance(v, float)},
            }
            history.append(row)

            score = val[self.cfg.monitor]
            improved = score > best_score if higher_is_better else score < best_score
            logger.info(
                "epoch %d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f macro-F1 %.3f%s",
                epoch, self.cfg.epochs, train_loss, train_acc,
                val["loss"], val["accuracy"], val["macro_f1"], "  <- best" if improved else "",
            )
            if improved:
                best_score = score
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint(self.out_dir / BEST_CHECKPOINT, epoch, val)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.cfg.early_stopping_patience:
                    logger.info("Early stopping: no improvement for %d epochs", epochs_without_improvement)
                    break

        history_df = pd.DataFrame(history)
        history_df.to_csv(self.out_dir / "history.csv", index=False)
        self.load_checkpoint(self.out_dir / BEST_CHECKPOINT)
        return history_df

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(progress):
            if self.cfg.max_batches_per_epoch and step >= self.cfg.max_batches_per_epoch:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
            progress.set_postfix(loss=f"{total_loss / seen:.4f}")
        return total_loss / max(seen, 1), correct / max(seen, 1)

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> dict[str, np.ndarray | float]:
        self.model.eval()
        losses, probs, labels, ids = [], [], [], []
        for step, batch in enumerate(loader):
            if self.cfg.max_batches_per_epoch and step >= self.cfg.max_batches_per_epoch:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
            losses.append(self.criterion(logits.float(), targets).item() * len(targets))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.append(batch["label"].numpy())
            ids.append(batch["image_id"].numpy())
        y_true = np.concatenate(labels)
        return {
            "y_true": y_true,
            "y_prob": np.concatenate(probs),
            "image_id": np.concatenate(ids),
            "loss": float(sum(losses) / len(y_true)),
        }

    def evaluate(self, loader: DataLoader) -> dict:
        preds = self.predict(loader)
        metrics = compute_metrics(preds["y_true"], preds["y_prob"], self.class_names)
        metrics["loss"] = preds["loss"]
        return metrics

    # -------------------------------------------------------------- checkpoints
    def save_checkpoint(self, path: Path, epoch: int, val_metrics: dict) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "class_names": list(self.class_names),
                **self.checkpoint_extra,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        return checkpoint
