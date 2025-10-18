import argparse
import os
from pathlib import Path

import torch


def save_empty_pt(target_path: str) -> None:
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([], str(path))
    print(f"Saved empty list to: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create valid .pt files containing an empty list []")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Absolute path(s) to .pt file(s) to create/overwrite with an empty list",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for p in args.paths:
        save_empty_pt(p)


if __name__ == "__main__":
    main()


