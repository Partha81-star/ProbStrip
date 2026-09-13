import argparse
import csv
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create leakage-resistant training manifests for supported medical datasets.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    mura = subparsers.add_parser("mura")
    mura.add_argument("dataset_dir")
    mura.add_argument("--output", default="mura_manifest.csv")
    chexpert = subparsers.add_parser("chexpert")
    chexpert.add_argument("dataset_dir")
    chexpert.add_argument("--label", default="Pleural Effusion")
    chexpert.add_argument("--uncertain", choices=("positive", "negative", "exclude"), default="positive")
    chexpert.add_argument("--output", default="chexpert_manifest.csv")
    return parser.parse_args()


def prepare_mura(root, output):
    root = Path(root).expanduser().resolve()
    path_files = [root / "train_image_paths.csv", root / "valid_image_paths.csv"]
    rows = []
    for path_file in path_files:
        if not path_file.is_file():
            continue
        with path_file.open("r", encoding="utf-8-sig", newline="") as handle:
            for source in csv.reader(handle):
                if not source:
                    continue
                relative_path = source[0].replace("\\", "/")
                relative_path = re.sub(r"^MURA-v1\.1/", "", relative_path, flags=re.I)
                patient = re.search(r"(?i)(patient\d+)", relative_path)
                rows.append(
                    {
                        "sample_id": relative_path.replace("/", "__"),
                        "relative_path": relative_path,
                        "patient_id": patient.group(1).lower() if patient else relative_path,
                        "abnormal": 1 if "positive" in relative_path.lower() else 0,
                    }
                )
    if not rows:
        raise FileNotFoundError("MURA train_image_paths.csv and valid_image_paths.csv were not found")
    output = Path(output)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} MURA image rows to {output}")


def prepare_chexpert(root, label, uncertain, output):
    root = Path(root).expanduser().resolve()
    source = root / "train.csv"
    if not source.is_file():
        raise FileNotFoundError(f"CheXpert train.csv was not found at {source}")
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get(label, "")
            if value in (None, ""):
                value = "0"
            if float(value) == -1:
                if uncertain == "exclude":
                    continue
                value = "1" if uncertain == "positive" else "0"
            relative_path = row["Path"].replace("\\", "/")
            relative_path = re.sub(r"^CheXpert-v1\.0(?:-small)?/", "", relative_path, flags=re.I)
            patient = re.search(r"(?i)(patient\d+)", relative_path)
            rows.append(
                {
                    "sample_id": relative_path.replace("/", "__"),
                    "relative_path": relative_path,
                    "patient_id": patient.group(1).lower() if patient else relative_path,
                    "target": int(float(value) > 0),
                    "label_name": label,
                }
            )
    output = Path(output)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} CheXpert rows for '{label}' to {output}")


def main(args):
    if args.dataset == "mura":
        prepare_mura(args.dataset_dir, args.output)
    else:
        prepare_chexpert(args.dataset_dir, args.label, args.uncertain, args.output)


if __name__ == "__main__":
    main(parse_args())
