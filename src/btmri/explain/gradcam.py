"""Grad-CAM: which regions of the image drove the model's decision?

Idea (Selvaraju et al., 2017): the last convolutional layer produces C feature maps, each a
coarse (e.g. 7x7) grid saying "pattern c is present here". We ask how much the score of the
target class would change if each feature map got stronger (its gradient), average that
into one importance weight per map, take the weighted sum of the maps, and keep only the
positive part. Upsampled to image size, this gives a heatmap of class evidence.

This implementation relies on timm's ``forward_features`` (image -> last feature maps) and
``forward_head`` (feature maps -> class scores) split, so it works for every timm CNN
without hooks or per-architecture layer names. Vision Transformers return token sequences
instead of 2-D maps and would need a reshape step; they are rejected explicitly for now.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module):
        if not (hasattr(model, "forward_features") and hasattr(model, "forward_head")):
            raise TypeError("GradCAM expects a timm model with forward_features/forward_head")
        self.model = model.eval()

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
            features = self.model.forward_features(images)
            if features.ndim != 4:
                raise ValueError(
                    f"Expected (B, C, h, w) feature maps, got shape {tuple(features.shape)}. "
                    "Transformer models need a token-to-grid reshape (not implemented yet)."
                )
            logits = self.model.forward_head(features)
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
