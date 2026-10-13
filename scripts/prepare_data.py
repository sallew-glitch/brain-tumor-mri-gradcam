"""Step 1: convert the raw .mat files into PNGs + metadata.csv and print dataset statistics.

Usage:
    python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from btmri.constants import CLASS_NAMES
from btmri.data.convert import convert_dataset
from btmri.viz import plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", default="data/raw", help="folder with mat/ and cvind.mat")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--size", type=int, default=256, help="output image side length in pixels")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    meta = convert_dataset(args.raw_dir, out_dir, args.size)

    print(f"\n{len(meta)} images from {meta['patient_id'].nunique()} patients")
    print("\nImages per class:\n" + meta["class_name"].value_counts().to_string())
    print("\nImages per official fold and class:")
    print(meta.pivot_table(index="cv_fold", columns="class_name", values="image_id", aggfunc="count"))
    print(f"\nTumor area: median {meta['tumor_area_frac'].median():.1%} of the image")

    # Visual sanity check: 4 random slices per class with the tumor outlined in green.
    sample = meta.groupby("label").sample(4, random_state=0)
    images = [np.asarray(Image.open(out_dir / p)) for p in sample["image_path"]]
    masks = [np.asarray(Image.open(out_dir / p)) > 127 for p in sample["mask_path"]]
    titles = [f"{CLASS_NAMES[r.label]} (patient {r.patient_id})" for r in sample.itertuples()]
    plot_image_grid(images, titles, out_dir / "samples.png", masks=masks)
    print(f"\nSample grid saved to {out_dir / 'samples.png'}")


if __name__ == "__main__":
    main()
