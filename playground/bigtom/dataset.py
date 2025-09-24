#!/usr/bin/env python3
"""
Simple CSV splitter: split a semicolon-delimited CSV into train/test subsets.

Defaults:
- input: data/bigtom/bigtom.csv (relative to this script)
- output_dir: data/splits (will create if missing)
- train_ratio: 0.8 (80/20 split)
- seed: 42 (reproducible shuffle)

Writes:
- output_dir/train.csv
- output_dir/test.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def split_csv(
    input_path: Path, output_dir: Path, train_ratio: float = 0.8, seed: int = 42
) -> tuple[int, int]:
    # Read all non-empty lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(lines)

    n = len(lines)
    n_train = int(round(train_ratio * n))
    train_lines = lines[:n_train]
    test_lines = lines[n_train:]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write out preserving content and adding newline if missing
    train_file = output_dir / "train.csv"
    test_file = output_dir / "test.csv"

    with open(train_file, "w", encoding="utf-8") as f:
        for ln in train_lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for ln in test_lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")

    return len(train_lines), len(test_lines)


def peek_csv(input_path: Path, max_data_rows: int = 1) -> None:
    """Print header candidate and first data row(s) with indexed columns."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    if not lines:
        print(f"[peek] File is empty: {input_path}")
        return

    def show_row(tag: str, row_text: str) -> None:
        cols = row_text.split(";")
        print(f"[{tag}] columns={len(cols)}")
        for idx, col in enumerate(cols):
            print(f"  [{idx:02d}] {col}")

    # Header candidate: first non-empty line
    show_row("header", lines[0])

    # First data rows (if available)
    data_rows = lines[1 : 1 + max(1, int(max_data_rows))]
    for i, row in enumerate(data_rows, start=1):
        show_row(f"data#{i}", row)


def pairwise_split_see_no(
    input_path: Path, output_dir: Path, seed: int = 42
) -> tuple[int, int]:
    """For each row with see/no scenarios, randomly put one scenario in train and the other in test.

    Output files:
    - output_dir/train_pair.csv  (only the chosen scenario per row; the other scenario column blanked)
    - output_dir/test_pair.csv   (the complementary scenario per row; the chosen scenario column blanked)
    """
    rng = random.Random(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]

    train_rows: list[str] = []
    test_rows: list[str] = []

    for ln in lines:
        cols = ln.split(";")
        # Ensure at least 14 columns to safely access indices used in pipeline
        if len(cols) < 14:
            cols = cols + [""] * (14 - len(cols))

        # Column indices based on evaluate_cbn.py
        # 0: story, 1: see observation, 2: no observation, 3: optA, 4: optB, 7: q_action,
        # 10: will_see, 13: will_no
        see_obs = cols[1].strip() if len(cols) > 1 else ""
        no_obs = cols[2].strip() if len(cols) > 2 else ""

        # Decide which scenario goes to train
        if see_obs and no_obs:
            train_choice = rng.choice(["see", "no"])
        elif see_obs:
            train_choice = "see"
        elif no_obs:
            train_choice = "no"
        else:
            # No scenario info; place in train by default unchanged
            train_rows.append(ln)
            continue

        # Build train line (blank the opposite scenario)
        tcols = cols[:]
        if train_choice == "see":
            # Keep see, blank no
            if len(tcols) > 2:
                tcols[2] = ""
            # For safety, blank will_no as well
            if len(tcols) > 13:
                tcols[13] = ""
        else:
            # Keep no, blank see
            if len(tcols) > 1:
                tcols[1] = ""
            if len(tcols) > 10:
                tcols[10] = ""
        train_rows.append(";".join(tcols))

        # Build test line as complementary scenario only if both existed
        if see_obs and no_obs:
            ucols = cols[:]
            if train_choice == "see":
                # Test keeps no, blank see
                if len(ucols) > 1:
                    ucols[1] = ""
                if len(ucols) > 10:
                    ucols[10] = ""
            else:
                # Test keeps see, blank no
                if len(ucols) > 2:
                    ucols[2] = ""
                if len(ucols) > 13:
                    ucols[13] = ""
            test_rows.append(";".join(ucols))

    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train_pair.csv"
    test_file = output_dir / "test_pair.csv"

    with open(train_file, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(row + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(row + "\n")

    return len(train_rows), len(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split CSV into train/test sets")
    parser.add_argument(
        "--input",
        default="data/bigtom/bigtom.csv",
        help="Path to input CSV (relative to script dir)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits",
        help="Output directory for train/test CSVs (relative to script dir)",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=0,
        help="Print header and first N data rows, then exit (no splitting)",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Pairwise split per row: randomly put see/no into different sets",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of rows to place in train set (0..1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_path = base_dir / args.input
    output_dir = base_dir / args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.peek and args.peek > 0:
        peek_csv(input_path, args.peek)
        return

    if args.pairwise:
        train_n, test_n = pairwise_split_see_no(input_path, output_dir, args.seed)
        print(f"Wrote {train_n} rows to {output_dir / 'train_pair.csv'}")
        print(f"Wrote {test_n} rows to {output_dir / 'test_pair.csv'}")
    else:
        train_n, test_n = split_csv(input_path, output_dir, args.train_ratio, args.seed)
        print(f"Wrote {train_n} rows to {output_dir / 'train.csv'}")
        print(f"Wrote {test_n} rows to {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
