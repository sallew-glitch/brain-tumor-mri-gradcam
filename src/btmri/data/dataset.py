"""PyTorch ``Dataset`` and ``DataLoader`` construction."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from btmri.config import ExperimentConfig
from btmri.data.convert import load_metadata
from btmri.data.splits import FoldSplit, get_fold_split, make_patient_folds
from btmri.data.transforms import build_eval_transform, build_train_transform


class BrainTumorDataset(Dataset):
    """Yields dicts so new fields (masks, metadata, ...) can be added without breaking callers.

    Each item has ``image`` (3xHxW float tensor), ``label`` (int) and ``image_id`` (int),
    plus ``mask`` (HxW float tensor in {0, 1}) when ``return_mask=True``. Masks are only
    meaningful with the deterministic eval transform, which is a plain square resize.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        root: str | Path,
        transform: Callable,
        return_mask: bool = False,
        mask_size: int | None = None,
    ):
        if return_mask and mask_size is None:
            raise ValueError("mask_size is required when return_mask=True")
        self.metadata = metadata.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.return_mask = return_mask
        self.mask_size = mask_size

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]
        image = Image.open(self.root / row["image_path"]).convert("L")
        item = {
            "image": self.transform(image),
            "label": int(row["label"]),
            "image_id": int(row["image_id"]),
        }
        if self.return_mask:
            mask = Image.open(self.root / row["mask_path"]).convert("L")
            mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
            item["mask"] = torch.from_numpy(np.asarray(mask) > 127).float()
        return item


def load_split(cfg: ExperimentConfig, test_fold: int) -> FoldSplit:
    metadata = load_metadata(cfg.data.processed_dir)
    fold_column = "cv_fold"
    if cfg.data.fold_source == "stratified_group":
        metadata["custom_fold"] = make_patient_folds(metadata, cfg.data.n_folds, cfg.seed)
        fold_column = "custom_fold"
    elif cfg.data.fold_source != "official":
        raise ValueError(f"Unknown data.fold_source '{cfg.data.fold_source}'")
    return get_fold_split(metadata, test_fold, cfg.data.n_folds, fold_column)


def make_loader(
    dataset: Dataset, cfg: ExperimentConfig, shuffle: bool, device: torch.device
) -> DataLoader:
    workers = cfg.data.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        # Each worker is a full Python+PyTorch process (~1 GB on Windows), so only the training
        # loader, which runs every epoch, keeps its workers alive between epochs.
        persistent_workers=workers > 0 and shuffle,
        pin_memory=device.type == "cuda",
    )


def build_dataloaders(
    cfg: ExperimentConfig, split: FoldSplit, device: torch.device
) -> dict[str, DataLoader]:
    root, size = cfg.data.processed_dir, cfg.data.image_size
    train_tf = build_train_transform(size, cfg.augment)
    eval_tf = build_eval_transform(size)
    return {
        "train": make_loader(BrainTumorDataset(split.train, root, train_tf), cfg, True, device),
        "val": make_loader(BrainTumorDataset(split.val, root, eval_tf), cfg, False, device),
        "test": make_loader(BrainTumorDataset(split.test, root, eval_tf), cfg, False, device),
    }
