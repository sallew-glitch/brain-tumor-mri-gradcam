"""Model construction.

Transfer learning: we start from a network already trained on ImageNet (1.2M everyday
photos). Its early layers detect generic edges, textures and shapes, which are useful for
MRI too. We replace its final 1000-class layer (the "head") with a new 3-class layer and
fine-tune. This needs far less data than training from scratch.

All models come from ``timm``, so any architecture is available by changing
``model.name`` in the config; no code changes are needed.
"""

from __future__ import annotations

import timm
import torch.nn as nn

from btmri.config import ModelConfig


def build_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    return timm.create_model(
        cfg.name,
        pretrained=cfg.pretrained,
        num_classes=num_classes,
        drop_rate=cfg.drop_rate,
    )


def _head_parameter_ids(model: nn.Module) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the classification head."""
    head_ids = _head_parameter_ids(model)
    for param in model.parameters():
        if id(param) not in head_ids:
            param.requires_grad = trainable


def parameter_groups(model: nn.Module, lr: float, backbone_lr_mult: float) -> list[dict]:
    """Give the pretrained backbone a smaller learning rate than the freshly initialised head,
    so fine-tuning adjusts the pretrained features gently instead of overwriting them."""
    head_ids = _head_parameter_ids(model)
    head = [p for p in model.parameters() if id(p) in head_ids]
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr * backbone_lr_mult, "name": "backbone"},
        {"params": head, "lr": lr, "name": "head"},
    ]
