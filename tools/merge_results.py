#!/usr/bin/env python3
"""Merge CSV results in output/results/ that share the same configuration.

Files are named like: {phase}_{config}.csv  (e.g. encoder_CIM_complex.csv, decoder_CIM_complex.csv)
This script groups files by config, sums energy/cycle/latency columns,
takes max for SRAM columns, and recalculates power columns.

Usage:
    python tools/merge_results.py [--input_dir output/results] [--output_dir output/merged]
"""

import argparse
import csv
import os
import re
from collections import defaultdict

# Columns that should be summed
SUM_COLS = {
    "Cycle", "Latency",
    "E_Compute", "E_DRAM_Access", "E_NOC_Hop",
    "E_SRAM_Read", "E_SRAM_Write", "E_SRAM_Access",
    "E_CIM_Local_Read", "Total_Energy",
}

# Columns that should take max
MAX_COLS = {
    "Max_Core_SRAM_Usage", "Peak_Total_SRAM_Usage",
}

# Columns kept as-is (must be identical across merged files)
KEEP_COLS = {
    "Tech", "Arch", "Batch Size", "Micro Batch Size", "DP",
    "Max_Core_SRAM_Core_Id",
}

# Power columns are recalculated from energy / (Cycle / freq)
# P_X = E_X * DP / (Cycle / freq)
POWER_MAP = {
    "P_Compute":        "E_Compute",
    "P_DRAM_Access":    "E_DRAM_Access",
    "P_NOC_Hop":        "E_NOC_Hop",
    "P_SRAM_Read":      "E_SRAM_Read",
    "P_SRAM_Write":     "E_SRAM_Write",
    "P_SRAM_Access":    "E_SRAM_Access",
    "P_CIM_Local_Read": "E_CIM_Local_Read",
    "P_Total_Power":    "Total_Energy",
}


def parse_config(filename):
    """Extract config key from filename like 'encoder_CIM_complex.csv' -> 'CIM_complex'."""
    name = os.path.splitext(filename)[0]
    # Remove the leading phase (encoder/decoder/etc.) prefix
    parts = name.split("_", 1)
    if len(parts) < 2:
        return None
    return parts[1]


def read_csv_row(filepath):
    """Read single-row CSV and return (header, row_dict)."""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None, None
        return reader.fieldnames, rows[0]


def merge_rows(rows, fieldnames):
    """Merge multiple row dicts into one."""
    merged = {}
    for col in fieldnames:
        values = [float(r[col]) if r[col] and col not in KEEP_COLS else r[col] for r in rows]

        if col in KEEP_COLS:
            merged[col] = values[0]
        elif col in SUM_COLS:
            merged[col] = sum(values)
        elif col in MAX_COLS:
            merged[col] = max(values)
        elif col in POWER_MAP:
            merged[col] = 0  # placeholder, recalculated below
        else:
            # Unknown column: sum by default
            merged[col] = sum(values)

    # Recalculate power: P = E * DP / (Cycle / freq)
    # From the CSV: Latency = Cycle / freq, so freq = Cycle / Latency
    cycle = merged["Cycle"]
    latency = merged["Latency"]
    dp = float(merged["DP"])
    if cycle > 0 and latency > 0:
        freq = cycle / latency  # in Hz (actually GHz * 1e9 but ratio works)
        for p_col, e_col in POWER_MAP.items():
            if p_col in merged and e_col in merged:
                merged[p_col] = merged[e_col] * dp / (cycle / freq)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge CSV results with same config")
    parser.add_argument("--input_dir", default="output/results", help="Input directory")
    parser.add_argument("--output_dir", default="output/merged", help="Output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: input directory '{args.input_dir}' not found")
        return

    # Group files by config
    groups = defaultdict(list)
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(".csv"):
            continue
        config = parse_config(fname)
        if config:
            groups[config].append(fname)

    if not groups:
        print("No CSV files found")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for config, files in sorted(groups.items()):
        if len(files) < 2:
            print(f"[skip] {config}: only 1 file ({files[0]}), copying as-is")
            # Still copy single files
            fieldnames, row = read_csv_row(os.path.join(args.input_dir, files[0]))
            if row is None:
                continue
            out_path = os.path.join(args.output_dir, f"{config}.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            print(f"  -> {out_path}")
            continue

        # Read all rows
        all_rows = []
        fieldnames = None
        phases = []
        for fname in files:
            fpath = os.path.join(args.input_dir, fname)
            hdr, row = read_csv_row(fpath)
            if row is None:
                continue
            if fieldnames is None:
                fieldnames = hdr
            all_rows.append(row)
            phases.append(fname.split("_", 1)[0])

        if len(all_rows) < 2:
            continue

        merged = merge_rows(all_rows, fieldnames)

        out_path = os.path.join(args.output_dir, f"{config}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(merged)

        print(f"[merged] {config}: {' + '.join(phases)} ({len(files)} files) -> {out_path}")


if __name__ == "__main__":
    main()
