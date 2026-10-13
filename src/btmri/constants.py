"""Facts about the dataset that the rest of the code relies on."""

# Order matches the dataset's own labels (1, 2, 3) shifted to 0-based indices.
CLASS_NAMES: tuple[str, ...] = ("meningioma", "glioma", "pituitary")
NUM_CLASSES = len(CLASS_NAMES)

# Pretrained ImageNet models expect inputs normalised with these statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

METADATA_FILENAME = "metadata.csv"
