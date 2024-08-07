#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import csv
from pathlib import Path

def find_zarr_directories(directory):
    return [os.path.join(directory, d) for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d)) and d.endswith('.zarr')]

def shuffle_and_split(directories, split_ratio=0.8):
    random.shuffle(directories)
    split_index = int(len(directories) * split_ratio)
    return directories[:split_index], directories[split_index:]

def export_to_csv(directories, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['directory_path'])
        for directory in directories:
            writer.writerow([directory])

def main(directory, split_ratio, output_dir):
    zarr_directories = find_zarr_directories(directory)

    if not zarr_directories:
        print(f"No .zarr directories found in {directory}")
        return

    train_directories, eval_directories = shuffle_and_split(zarr_directories, split_ratio)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_to_csv(train_directories, output_dir / 'train.csv')
    export_to_csv(eval_directories, output_dir / 'eval.csv')

    print(f"Processed {len(zarr_directories)} .zarr directories")
    print(f"Exported {len(train_directories)} directories to {output_dir / 'train.csv'}")
    print(f"Exported {len(eval_directories)} directories to {output_dir / 'eval.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .zarr directories and create train/eval CSV files")
    parser.add_argument("directory", help="Directory to search for .zarr directories")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/eval split ratio (default: 0.8)")
    parser.add_argument("--output-dir", default=".", help="Output directory for CSV files (default: current directory)")

    args = parser.parse_args()

    main(args.directory, args.split_ratio, args.output_dir)
