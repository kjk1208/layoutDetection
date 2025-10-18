import argparse
import os
from pathlib import Path
import pandas as pd


def list_files(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {f.name for f in directory.iterdir() if f.is_file()}


def make_keys(files: set[str]) -> set[str]:
    return {os.path.splitext(f)[0] for f in files}


def main():
    parser = argparse.ArgumentParser(description="Check train.csv rows against actual files; print missing entries only (no writes).")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root to dataset, e.g., /home/kjk/movers/PosterO-CVPR2025/RALF/DATA")
    parser.add_argument("--dataset", type=str, default="pku", help="Dataset name under dataset_root (e.g., pku)")
    parser.add_argument("--csv", type=str, default="train.csv", help="Annotation CSV filename (e.g., train.csv)")
    parser.add_argument("--image_split", type=str, default="train", help="Image split folder name (train/test/etc)")
    parser.add_argument("--saliency_dir", type=str, default="saliency_sub", help="Saliency subfolder name")
    parser.add_argument("--saliency_suffix", type=str, default="_mask_pred", help="Suffix before extension in saliency filenames")
    args = parser.parse_args()

    base = Path(args.dataset_root) / args.dataset
    img_root = base / "image" / args.image_split
    csv_path = base / "annotation" / args.csv

    input_dir = img_root / "input"
    original_dir = img_root / "original"
    closedm_dir = img_root / "closedm"
    saliency_dir = img_root / args.saliency_dir

    # List files and build key sets
    input_keys = make_keys(list_files(input_dir))
    original_keys = make_keys(list_files(original_dir))
    closedm_keys = make_keys(list_files(closedm_dir))
    saliency_keys = set()
    for f in list_files(saliency_dir):
        stem, _ = os.path.splitext(f)
        if stem.endswith(args.saliency_suffix):
            stem = stem[: -len(args.saliency_suffix)]
        saliency_keys.add(stem)

    # Load CSV
    df = pd.read_csv(csv_path)
    stems = df["poster_path"].astype(str).str.replace(".png", "", regex=False).str.replace(".jpg", "", regex=False)

    missing_input = sorted(set(stems) - input_keys)
    missing_original = sorted(set(stems) - original_keys)
    missing_closedm = sorted(set(stems) - closedm_keys)
    missing_saliency = sorted(set(stems) - saliency_keys)

    print(f"CSV: {csv_path}")
    print(f"Dirs -> input: {input_dir}, original: {original_dir}, closedm: {closedm_dir}, saliency: {saliency_dir} (suffix='{args.saliency_suffix}')")
    print("")
    print(f"Total CSV rows: {len(df)} (unique poster_path: {len(set(stems))})")
    print(f"Missing in input: {len(missing_input)}")
    print(f"Missing in original: {len(missing_original)}")
    print(f"Missing in closedm: {len(missing_closedm)}")
    print(f"Missing in saliency: {len(missing_saliency)}")

    def preview(name: str, items: list[str], limit: int = 50):
        if not items:
            print(f"\n[{name}] none missing")
            return
        print(f"\n[{name}] first {min(limit, len(items))} of {len(items)}:")
        for k in items[:limit]:
            if name == "saliency":
                print(f"{k}{args.saliency_suffix}.png")
            else:
                print(f"{k}.png")

    preview("input", missing_input)
    preview("original", missing_original)
    preview("closedm", missing_closedm)
    preview("saliency", missing_saliency)


if __name__ == "__main__":
    main()


