import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Split CSV into train/test by poster_path groups (keep groups intact).")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV path (e.g., .../annotation/train.csv)")
    parser.add_argument("--train_out", type=str, default=None, help="Output CSV for train (default: <input_dir>/train_split.csv)")
    parser.add_argument("--test_out", type=str, default=None, help="Output CSV for test (default: <input_dir>/test_split.csv)")
    parser.add_argument("--ratio", type=str, default="9:1", help="Train:Test ratio as 'a:b' (default 9:1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling groups")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    out_dir = input_path.parent
    train_out = Path(args.train_out) if args.train_out else out_dir / "train_split.csv"
    test_out = Path(args.test_out) if args.test_out else out_dir / "test_split.csv"

    # Parse ratio
    try:
        a, b = args.ratio.split(":")
        a, b = int(a), int(b)
        assert a > 0 and b > 0
    except Exception:
        raise ValueError("--ratio must be in the form 'a:b' with positive integers, e.g., 9:1")
    frac_test = b / (a + b)

    # Load CSV
    df = pd.read_csv(input_path)
    if "poster_path" not in df.columns:
        raise ValueError("Input CSV must contain 'poster_path' column")

    # Unique groups by poster_path
    poster_groups = df.groupby("poster_path").indices
    posters = np.array(list(poster_groups.keys()))

    # Shuffle and split groups
    rng = np.random.default_rng(args.seed)
    rng.shuffle(posters)
    num_test = max(1, int(round(len(posters) * frac_test)))
    test_posters = set(posters[:num_test])

    test_mask = df["poster_path"].isin(test_posters)
    df_test = df[test_mask].reset_index(drop=True)
    df_train = df[~test_mask].reset_index(drop=True)

    # Save
    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)

    # Report
    print(f"Input: {input_path}")
    print(f"Total posters: {len(posters)} | rows: {len(df)}")
    print(f"Train posters: {len(set(df_train.poster_path))} | rows: {len(df_train)} -> {train_out}")
    print(f"Test posters: {len(set(df_test.poster_path))} | rows: {len(df_test)} -> {test_out}")


if __name__ == "__main__":
    main()


