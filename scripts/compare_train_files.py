import argparse
import os
from pathlib import Path


def list_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return [f.name for f in directory.iterdir() if f.is_file()]


def to_key(filename: str, is_saliency: bool, saliency_suffix: str) -> str:
    stem, _ = os.path.splitext(filename)
    if is_saliency and stem.endswith(saliency_suffix):
        stem = stem[: -len(saliency_suffix)]
    return stem


def main():
    parser = argparse.ArgumentParser(description="Compare file presence across train subfolders and report mismatches.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(Path.home() / "kjk/movers/PosterO-CVPR2025/RALF/DATA/pku/image/train"),
        help="Base train directory containing subfolders: closedm, input, original, saliency_sub",
    )
    parser.add_argument("--closedm", type=str, default="closedm", help="Closedm folder name")
    parser.add_argument("--input", dest="input_dir", type=str, default="input", help="Input folder name")
    parser.add_argument("--original", type=str, default="original", help="Original folder name")
    parser.add_argument("--saliency", type=str, default="saliency_sub", help="Saliency folder name")
    parser.add_argument(
        "--saliency_suffix",
        type=str,
        default="_mask_pred",
        help="Suffix appended to saliency filenames before extension (e.g., _mask_pred)",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="train_folder_mismatches.txt",
        help="Output report filename (written under base_dir)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    paths = {
        "closedm": base_dir / args.closedm,
        "input": base_dir / args.input_dir,
        "original": base_dir / args.original,
        "saliency": base_dir / args.saliency,
    }

    files = {name: list_files(path) for name, path in paths.items()}

    # Build canonical key sets for comparison
    key_sets = {
        "closedm": {to_key(f, False, args.saliency_suffix) for f in files["closedm"]},
        "input": {to_key(f, False, args.saliency_suffix) for f in files["input"]},
        "original": {to_key(f, False, args.saliency_suffix) for f in files["original"]},
        "saliency": {to_key(f, True, args.saliency_suffix) for f in files["saliency"]},
    }

    union_keys = set().union(*key_sets.values())

    def expected_name(folder: str, key: str) -> str:
        if folder == "saliency":
            return f"{key}{args.saliency_suffix}.png"
        else:
            # We do not enforce extension differences: default to .png in report
            return f"{key}.png"

    # Compute missing keys per folder (relative to union)
    missing = {name: sorted(union_keys - keys) for name, keys in key_sets.items()}

    # Extra files that have keys not present in any other folder (optional but useful)
    extras = {}
    for name, keys in key_sets.items():
        others = union_keys - keys
        # Extras: files present in this folder whose key is absent from the intersection of other folders
        others_intersection = set.intersection(*[ks for n, ks in key_sets.items() if n != name]) if len(key_sets) > 1 else set()
        extras[name] = sorted(keys - others_intersection)

    report_path = base_dir / args.report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Base directory: {base_dir}\n")
        for name, path in paths.items():
            f.write(f"{name}: {path} (files={len(files[name])}, keys={len(key_sets[name])})\n")
        f.write("\n")

        f.write("Missing files by folder (expected names):\n")
        for name in ["closedm", "input", "original", "saliency"]:
            f.write(f"\n[{name}] missing: {len(missing[name])}\n")
            for key in missing[name]:
                f.write(expected_name(name, key) + "\n")

        f.write("\n")
        f.write("Extras present by folder (keys that may not exist elsewhere):\n")
        for name in ["closedm", "input", "original", "saliency"]:
            f.write(f"\n[{name}] extras (by key): {len(extras[name])}\n")
            for key in extras[name]:
                f.write(key + "\n")

    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()


