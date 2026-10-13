"""Grad-CAM: which regions of the image drove the model's decision?

Idea (Selvaraju et al., 2017): a convolutional layer produces C feature maps, each a grid
saying "pattern c is present here". We ask how much the score of the target class would
change if each feature map got stronger (its gradient), average that into one importance
weight per map, take the weighted sum of the maps, and keep only the positive part.
Upsampled to image size, this gives a heatmap of class evidence.

Which layer? The heatmap is only as detailed as the layer's grid:

* ``"last"`` (default): the final feature maps. Every common CNN gives a coarse 7x7 grid here
  at 224px input, so each cell covers 32x32 pixels (bigger than a typical tumor).
* ``"penultimate"``: the stage before, a 14x14 grid. Finer, but a bit less "class-specific".
* any module name, e.g. ``"layer3"`` (ResNet) or ``"blocks.4"`` (EfficientNet).

``"last"`` uses timm's ``forward_features``/``forward_head`` split; the other options look
the layer up in timm's ``feature_info``, so all of them work for every timm CNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_layer(model: nn.Module, layer: str | None) -> str | None:
    """Turn a layer option into a module name (None means the final feature maps)."""
    if layer in (None, "last"):
        return None
    if layer == "penultimate":
        return model.feature_info[-2]["module"]
    model.get_submodule(layer)  # raises AttributeError with a clear message if it doesn't exist
    return layer


class GradCAM:
    def __init__(self, model: nn.Module, layer: str | None = None):
        if not (hasattr(model, "forward_features") and hasattr(model, "forward_head")):
            raise TypeError("GradCAM expects a timm model with forward_features/forward_head")
        self.model = model.eval()
        self.layer_name = resolve_layer(model, layer)

    def _features_and_logits(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_name is None:
            features = self.model.forward_features(images)
            return features, self.model.forward_head(features)

        captured = {}
        handle = self.model.get_submodule(self.layer_name).register_forward_hook(
            lambda _module, _inputs, output: captured.update(features=output)
        )
        try:
            logits = self.model(images)
        finally:
            handle.remove()
        return captured["features"], logits

    def __call__(
        self, images: torch.Tensor, target_classes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cams, logits)``.

        ``cams`` has shape (B, H, W) with values in [0, 1], same H and W as ``images``.
        ``target_classes`` defaults to each image's predicted class.
        """
        with torch.enable_grad():
            # Making the input require grad guarantees a graph even if weights are frozen.
            images = images.detach().clone().requires_grad_(True)
            features, logits = self._features_and_logits(images)
            if features.ndim != 4:
                raise ValueError(
                    f"Expected (B, C, h, w) feature maps, got shape {tuple(features.shape)}. "
                    "Transformer models need a token-to-grid reshape (not implemented yet)."
                )
            if target_classes is None:
                target_classes = logits.argmax(dim=1)
            target_scores = logits.gather(1, target_classes.view(-1, 1)).sum()
            (gradients,) = torch.autograd.grad(target_scores, features)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = F.relu((weights * features).sum(dim=1, keepdim=True))  # (B, 1, h, w)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        flat = cam.flatten(1)
        low, high = flat.min(dim=1).values, flat.max(dim=1).values
        cam = (cam - low.view(-1, 1, 1)) / (high - low).clamp(min=1e-8).view(-1, 1, 1)
        return cam.detach(), logits.detach()
