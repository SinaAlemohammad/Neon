#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path

def build_label_index(root_dir: Path):
    """
    Walk root_dir, find files matching *_label{digit+}.png|jpg|jpeg,
    and return list of [relative_path, label].
    """
    entries = []
    pattern = re.compile(r'^(?P<base>.+)_label(?P<label>\d+)\.(?:png|jpe?g)$', re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            m = pattern.match(fname)
            if not m:
                continue
            label = int(m.group('label'))
            # compute relative path, using forward slashes
            rel_dir = os.path.relpath(dirpath, root_dir)
            rel_path = os.path.join(rel_dir, fname) if rel_dir != '.' else fname
            rel_path = rel_path.replace(os.path.sep, '/')
            entries.append([rel_path, label])

    # sort entries by path
    entries.sort(key=lambda x: x[0])
    return entries

def main():
    parser = argparse.ArgumentParser(
        description="Build dataset.json listing image paths and labels"
    )
    parser.add_argument(
        "--root", "-r",
        type=Path,
        required=True,
        help="Root directory containing your subdirs of images"
    )
    args = parser.parse_args()

    root_dir = args.root
    if not root_dir.is_dir():
        parser.error(f"{root_dir} is not a directory")

    labels = build_label_index(root_dir)
    out = {"labels": labels}

    # Save to dataset.json inside the same root directory
    dataset_path = root_dir / "dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(labels)} entries to {dataset_path}")

if __name__ == "__main__":
    main()
