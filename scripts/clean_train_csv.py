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
    parser = argparse.ArgumentParser(description="Clean train.csv by removing rows whose files are missing in train folders.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root to dataset, e.g., /home/kjk/movers/PosterO-CVPR2025/RALF/DATA")
    parser.add_argument("--dataset", type=str, default="pku", help="Dataset name under dataset_root (e.g., pku)")
    parser.add_argument("--csv", type=str, default="train.csv", help="Annotation CSV filename (e.g., train.csv)")
    parser.add_argument("--image_split", type=str, default="train", help="Image split folder name (train/test/etc)")
    parser.add_argument("--saliency_dir", type=str, default="saliency_sub", help="Saliency subfolder name")
    parser.add_argument("--saliency_suffix", type=str, default="_mask_pred", help="Suffix before extension in saliency filenames")
    parser.add_argument("--out_prefix", type=str, default="cleaned_", help="Prefix for cleaned CSV output")

    args = parser.parse_args()

    base = Path(args.dataset_root) / args.dataset
    img_root = base / "image" / args.image_split
    csv_path = base / "annotation" / args.csv

    input_dir = img_root / "input"
    original_dir = img_root / "original"
    closedm_dir = img_root / "closedm"
    saliency_dir = img_root / args.saliency_dir

    # List files
    input_files = list_files(input_dir)
    original_files = list_files(original_dir)
    closedm_files = list_files(closedm_dir)
    saliency_files = list_files(saliency_dir)

    # Build key sets (stem without extension). For saliency, strip suffix.
    input_keys = make_keys(input_files)
    original_keys = make_keys(original_files)
    closedm_keys = make_keys(closedm_files)
    # saliency: names like "123_mask_pred.png" -> key "123"
    saliency_keys = set()
    for f in saliency_files:
        stem, _ = os.path.splitext(f)
        if stem.endswith(args.saliency_suffix):
            stem = stem[: -len(args.saliency_suffix)]
        saliency_keys.add(stem)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Normalize poster_path to stems for comparison
    stems = df["poster_path"].astype(str).str.replace(".png", "", regex=False).str.replace(".jpg", "", regex=False)

    # Keep rows only if the key exists in ALL required folders
    keep_mask = (
        stems.isin(input_keys)
        & stems.isin(original_keys)
        & stems.isin(closedm_keys)
        & stems.isin(saliency_keys)
    )

    kept_df = df[keep_mask].reset_index(drop=True)
    dropped_df = df[~keep_mask].reset_index(drop=True)

    out_dir = csv_path.parent
    kept_out = out_dir / f"{args.out_prefix}{args.csv}"
    drop_out = out_dir / "dropped_rows.csv"

    kept_df.to_csv(kept_out, index=False)
    dropped_df.to_csv(drop_out, index=False)

    print(f"CSV: {csv_path}")
    print(f"Input dir: {input_dir} ({len(input_files)})")
    print(f"Original dir: {original_dir} ({len(original_files)})")
    print(f"Closedm dir: {closedm_dir} ({len(closedm_files)})")
    print(f"Saliency dir: {saliency_dir} ({len(saliency_files)})  suffix='{args.saliency_suffix}'")
    print(f"Kept rows: {len(kept_df)}  -> {kept_out}")
    print(f"Dropped rows: {len(dropped_df)}  -> {drop_out}")


if __name__ == "__main__":
    main()


