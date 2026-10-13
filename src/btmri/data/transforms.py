"""Image preprocessing and data augmentation.

Augmentation creates slightly altered copies of training images (small rotations, crops,
flips, brightness changes) so the model learns features that survive those changes instead
of memorising the exact training pixels. It is applied to training images only; validation
and test images are only resized and normalised.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2 as T

from btmri.config import AugmentConfig
from btmri.constants import IMAGENET_MEAN, IMAGENET_STD


def _to_model_input() -> list:
    return [
        # The MRI is grayscale but ImageNet models expect 3 channels: copy it into R, G and B.
        T.Grayscale(num_output_channels=3),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def build_eval_transform(image_size: int) -> T.Compose:
    return T.Compose([T.Resize((image_size, image_size), antialias=True), *_to_model_input()])


def build_train_transform(image_size: int, aug: AugmentConfig) -> T.Compose:
    if not aug.enabled:
        return build_eval_transform(image_size)
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size, scale=(aug.crop_scale_min, 1.0), ratio=(0.9, 1.1), antialias=True
            ),
            T.RandomHorizontalFlip(aug.hflip_p),
            T.RandomRotation(aug.rotation_deg),
            T.ColorJitter(brightness=aug.brightness, contrast=aug.contrast),
            *_to_model_input(),
        ]
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation, giving values in [0, 1] for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(-1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(-1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)
