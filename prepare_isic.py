import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Join ISIC labels and grouping metadata.")
    parser.add_argument("dataset_dir")
    return parser.parse_args()


def main(args):
    root = Path(args.dataset_dir).expanduser().resolve()
    labels_path = root / "ground_truth.csv"
    supplement_path = root / "supplement.csv"
    output_path = root / "training_manifest.csv"
    with supplement_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = {row["isic_id"]: row for row in csv.DictReader(handle)}
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["isic_id", "malignant", "lesion_id", "attribution", "copyright_license"],
        )
        writer.writeheader()
        for label in labels:
            sample_id = label["isic_id"]
            info = metadata.get(sample_id, {})
            writer.writerow(
                {
                    "isic_id": sample_id,
                    "malignant": label["malignant"],
                    "lesion_id": info.get("lesion_id") or sample_id,
                    "attribution": info.get("attribution", ""),
                    "copyright_license": info.get("copyright_license", ""),
                }
            )
    print(f"Wrote {len(labels)} labeled rows to {output_path}")


if __name__ == "__main__":
    main(parse_args())
