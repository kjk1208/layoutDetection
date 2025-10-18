import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Force 'split' column to 'test' for all rows in a CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/annotation/test.csv",
        help="Path to CSV to modify",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Save a backup file alongside the CSV (adds .bak suffix)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("CSV must contain a 'split' column")

    before_counts = df["split"].value_counts(dropna=False).to_dict()
    df["split"] = "test"
    after_counts = df["split"].value_counts(dropna=False).to_dict()

    if args.backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        df.to_csv(backup_path, index=False)

    df.to_csv(csv_path, index=False)

    print(f"Updated: {csv_path}")
    print(f"Before split counts: {before_counts}")
    print(f"After split counts:  {after_counts}")


if __name__ == "__main__":
    main()


