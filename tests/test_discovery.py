from pathlib import Path

from data.discovery import (
    dataset_manifest,
    discover_image_mask_pairs,
    group_aware_split,
    subject_group,
)


def test_discovery_matches_common_chase_mask_names(tmp_path):
    images = tmp_path / "Images"
    masks = tmp_path / "Masks"
    images.mkdir()
    masks.mkdir()
    (images / "Image_01L.jpg").touch()
    (images / "Image_01R.jpg").touch()
    (masks / "Image_01L_1stHO.png").touch()
    (masks / "Image_01R_1stHO.png").touch()

    pairs = discover_image_mask_pairs(tmp_path)

    assert len(pairs) == 2
    assert subject_group(pairs[0][0]) == "image_01"
    assert subject_group(pairs[1][0]) == "image_01"


def test_group_split_keeps_paired_eyes_together():
    pairs = [
        (Path(f"patient_{patient}{eye}.jpg"), Path(f"mask_{patient}{eye}.png"))
        for patient in range(1, 6)
        for eye in ("L", "R")
    ]

    train, validation = group_aware_split(pairs, validation_fraction=0.4, seed=7)
    train_groups = {subject_group(pair[0]) for pair in train}
    validation_groups = {subject_group(pair[0]) for pair in validation}

    assert train_groups
    assert validation_groups
    assert train_groups.isdisjoint(validation_groups)


def test_dataset_manifest_is_reproducible(tmp_path):
    images = tmp_path / "Images"
    masks = tmp_path / "Masks"
    images.mkdir()
    masks.mkdir()
    image = images / "Image_01L.jpg"
    mask = masks / "Image_01L_1stHO.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")

    pairs = discover_image_mask_pairs(tmp_path)
    first = dataset_manifest(pairs)
    second = dataset_manifest(pairs)

    assert first == second
    assert first["pair_count"] == 1
    assert len(first["pairs_sha256"]) == 64
