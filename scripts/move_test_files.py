import argparse
import os
from pathlib import Path
import pandas as pd
import shutil


def resolve_filename(base_dir: Path, stem: str, prefer_exts=(".png", ".jpg")) -> str | None:
    for ext in prefer_exts:
        candidate = base_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def move_file(src_dir: Path, dst_dir: Path, filename: str) -> bool:
    src = src_dir / filename
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename
    if not src.exists():
        return False
    shutil.move(str(src), str(dst))
    return True


def main():
    parser = argparse.ArgumentParser(description="Move files listed in test.csv from train/* to test/* across four subfolders.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root to dataset, e.g., /home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name under dataset_root (e.g., pku)")
    parser.add_argument("--csv", type=str, required=True, help="Path to test.csv (absolute path recommended)")
    parser.add_argument("--train_split", type=str, default="train", help="Train split folder name (default: train)")
    parser.add_argument("--test_split", type=str, default="test", help="Test split folder name (default: test)")
    parser.add_argument("--saliency_dir", type=str, default="saliency_sub", help="Saliency folder name")
    parser.add_argument("--saliency_suffix", type=str, default="_mask_pred", help="Saliency filename suffix before extension")
    parser.add_argument("--ext_order", type=str, default=".png,.jpg", help="Extension check order, comma-separated (default: .png,.jpg)")
    parser.add_argument("--dry_run", action="store_true", help="Do not move, only print planned actions")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    base_image = dataset_root / args.dataset / "image"
    train_root = base_image / args.train_split
    test_root = base_image / args.test_split

    dirs = {
        "input": (train_root / "input", test_root / "input"),
        "original": (train_root / "original", test_root / "original"),
        "closedm": (train_root / "closedm", test_root / "closedm"),
        "saliency": (train_root / args.saliency_dir, test_root / args.saliency_dir),
    }

    df = pd.read_csv(args.csv)
    stems = df["poster_path"].astype(str).str.replace(".png", "", regex=False).str.replace(".jpg", "", regex=False).unique()

    ext_order = tuple([e if e.startswith(".") else f".{e}" for e in args.ext_order.split(",")])

    moved_counts = {k: 0 for k in dirs}
    missing_counts = {k: 0 for k in dirs}

    print(f"Moving {len(stems)} posters from {train_root} -> {test_root}")
    for stem in stems:
        # Resolve filenames per folder
        # input, original, closedm: use detected extension
        for name in ["input", "original", "closedm"]:
            src_dir, dst_dir = dirs[name]
            filename = resolve_filename(src_dir, stem, prefer_exts=ext_order)
            if filename is None:
                missing_counts[name] += 1
                print(f"[MISS] {name}: {stem} (no file with {ext_order})")
                continue
            if args.dry_run:
                print(f"[PLAN] move {name}: {src_dir/filename} -> {dst_dir/filename}")
            else:
                if move_file(src_dir, dst_dir, filename):
                    moved_counts[name] += 1
                else:
                    missing_counts[name] += 1
                    print(f"[MISS] {name}: {src_dir/filename}")

        # saliency: add suffix before extension; try ext order
        name = "saliency"
        src_dir, dst_dir = dirs[name]
        filename = None
        for ext in ext_order:
            candidate = f"{stem}{args.saliency_suffix}{ext}"
            if (src_dir / candidate).exists():
                filename = candidate
                break
        if filename is None:
            missing_counts[name] += 1
            print(f"[MISS] {name}: {stem} (no saliency with suffix '{args.saliency_suffix}' and exts {ext_order})")
        else:
            if args.dry_run:
                print(f"[PLAN] move {name}: {src_dir/filename} -> {dst_dir/filename}")
            else:
                if move_file(src_dir, dst_dir, filename):
                    moved_counts[name] += 1
                else:
                    missing_counts[name] += 1
                    print(f"[MISS] {name}: {src_dir/filename}")

    print("\nSummary:")
    for name in ["input", "original", "closedm", "saliency"]:
        print(f"{name}: moved {moved_counts[name]}, missing {missing_counts[name]}")


if __name__ == "__main__":
    main()


