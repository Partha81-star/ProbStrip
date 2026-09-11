import random
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _find_named_directory(root: Path, name: str) -> Path:
    matches = [path for path in root.iterdir() if path.is_dir() and path.name.lower() == name]
    if not matches:
        raise FileNotFoundError(f"Expected a '{name}' directory inside {root}")
    return matches[0]


def discover_image_mask_pairs(dataset_dir):
    root = Path(dataset_dir).expanduser().resolve()
    images_dir = _find_named_directory(root, "images")
    masks_dir = _find_named_directory(root, "masks")
    masks_by_stem = {path.stem.lower(): path for path in masks_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    for image_path in sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    ):
        candidates = (
            f"{image_path.stem}_1stho",
            f"{image_path.stem}_2ndho",
            image_path.stem,
        )
        mask_path = next(
            (masks_by_stem[name.lower()] for name in candidates if name.lower() in masks_by_stem),
            None,
        )
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def subject_group(image_path) -> str:
    """Infer a conservative group key for common paired-eye filename conventions."""
    stem = Path(image_path).stem
    stem = re.sub(r"(?i)(?:[_-](?:left|right|od|os)|(?<=\d)[lr])$", "", stem)
    return stem.lower()


def group_aware_split(pairs, validation_fraction=0.2, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = {}
    for image_path, mask_path in pairs:
        groups.setdefault(subject_group(image_path), []).append((image_path, mask_path))
    if len(groups) < 2:
        raise ValueError("At least two inferred subject groups are required")

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    validation_count = max(1, round(len(group_names) * validation_fraction))
    validation_groups = set(group_names[:validation_count])
    train = []
    validation = []
    for group_name, group_pairs in groups.items():
        destination = validation if group_name in validation_groups else train
        destination.extend(group_pairs)
    if not train:
        moved_group = group_names[-1]
        train.extend(groups[moved_group])
        validation = [
            pair
            for pair in validation
            if subject_group(pair[0]) != moved_group
        ]
    return train, validation

