"""Experiment configuration.

Every experiment is fully described by one ``ExperimentConfig``. Configs live in YAML
files under ``configs/`` and are loaded into typed dataclasses, so that:

* typos are caught early (unknown keys raise an error instead of being silently ignored),
* editors can autocomplete ``cfg.train.lr`` etc.,
* the exact resolved config is saved next to every run for reproducibility.

A YAML file may inherit from another with ``_base_: other.yaml`` and only override the
keys it changes. Single values can also be overridden from the command line, e.g.
``--set train.epochs=3 model.name=efficientnet_b0``.
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    # "official": the patient-disjoint 5-fold split shipped with the dataset (cvind.mat).
    # "stratified_group": regenerate patient-disjoint folds with StratifiedGroupKFold(seed).
    fold_source: str = "official"
    n_folds: int = 5


@dataclass
class AugmentConfig:
    enabled: bool = True
    hflip_p: float = 0.5
    rotation_deg: float = 10.0
    crop_scale_min: float = 0.85
    brightness: float = 0.1
    contrast: float = 0.1


@dataclass
class ModelConfig:
    # Any timm model name works, e.g. resnet18, resnet50, efficientnet_b0, densenet121,
    # convnext_tiny. See https://huggingface.co/timm for the full list.
    name: str = "resnet18"
    pretrained: bool = True
    drop_rate: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 3e-4
    # Learning rate of the pretrained backbone relative to the new classifier head.
    backbone_lr_mult: float = 1.0
    weight_decay: float = 1e-4
    # Train only the new head for the first N epochs, then fine-tune everything.
    freeze_backbone_epochs: int = 1
    label_smoothing: float = 0.0
    # Weight the loss by inverse class frequency (glioma is ~2x more common).
    class_weighted_loss: bool = True
    # Metric on the validation fold used to pick the best checkpoint.
    monitor: str = "macro_f1"
    early_stopping_patience: int = 5
    # Mixed precision; only used on CUDA GPUs.
    amp: bool = True
    # Debug helper: cap the batches per epoch (None = use everything).
    max_batches_per_epoch: int | None = None


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | xpu | mps
    output_dir: str = "outputs/runs"
    # Which folds to use as the test fold. [1, 2, 3, 4, 5] = full cross-validation.
    folds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(cls: type, values: dict[str, Any]) -> Any:
    """Recursively build dataclass ``cls`` from a nested dict, rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}. Valid keys: {sorted(known)}")
        field_type = hints[key]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            value = _from_dict(field_type, value)
        kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_bases(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    base_name = raw.pop("_base_", None)
    if base_name is None:
        return raw
    base = _load_yaml_with_bases((path.parent / base_name).resolve())
    return _deep_merge(base, raw)


def apply_overrides(values: dict, overrides: list[str]) -> dict:
    """Apply ``section.key=value`` strings. Values are parsed as YAML (so 3 -> int, true -> bool)."""
    values = copy.deepcopy(values)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw_value = item.split("=", 1)
        *parents, leaf = dotted.strip().split(".")
        node = values
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw_value)
    return values


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config (or the defaults if ``path`` is None) and apply CLI overrides."""
    values = _load_yaml_with_bases(Path(path)) if path else {}
    values = apply_overrides(values, overrides or [])
    return _from_dict(ExperimentConfig, values)


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, values)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
