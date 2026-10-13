"""Plots saved alongside every run.

Uses Matplotlib's object-oriented ``Figure`` API rather than ``pyplot``, so it works the
same in scripts, on servers without a display, and inside notebooks.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_history(history: pd.DataFrame, path: str | Path) -> None:
    fig = Figure(figsize=(11, 4))
    ax_loss, ax_score = fig.subplots(1, 2)
    ax_loss.plot(history["epoch"], history["train_loss"], label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], label="validation")
    ax_loss.set(title="Loss", xlabel="epoch")
    ax_score.plot(history["epoch"], history["train_accuracy"], label="train accuracy")
    ax_score.plot(history["epoch"], history["val_accuracy"], label="val accuracy")
    ax_score.plot(history["epoch"], history["val_macro_f1"], label="val macro-F1")
    ax_score.set(title="Scores", xlabel="epoch", ylim=(0, 1.02))
    for ax in (ax_loss, ax_score):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_confusion_matrix(cm: Sequence[Sequence[int]], class_names: Sequence[str], path: str | Path) -> None:
    cm = np.asarray(cm)
    row_frac = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = Figure(figsize=(5, 4.4))
    ax = fig.subplots()
    ax.imshow(row_frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if row_frac[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{row_frac[i, j]:.0%}", ha="center", va="center", color=color)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="predicted", ylabel="true", title="Confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    path: str | Path,
    masks: Sequence[np.ndarray] | None = None,
    heatmaps: Sequence[np.ndarray] | None = None,
    cols: int = 4,
) -> None:
    """Grayscale images with optional tumor-mask outlines and Grad-CAM heatmap overlays."""
    rows = int(np.ceil(len(images) / cols))
    fig = Figure(figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(images):
            continue
        ax.imshow(images[i], cmap="gray")
        if heatmaps is not None:
            ax.imshow(heatmaps[i], cmap="jet", alpha=0.4, vmin=0, vmax=1)
        if masks is not None:
            ax.contour(masks[i], levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(titles[i], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
