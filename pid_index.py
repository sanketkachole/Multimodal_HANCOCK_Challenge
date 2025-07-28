#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pid_index.py
──────────────────
Reads the targets.csv file, extracts every unique patient_id,
assigns each one a consecutive integer row ID, and stores the
resulting dataframe to pid_index.pkl for reuse by all loaders.

USAGE
$ python build_pid_index.py \
      --targets_csv /N/project/wsiclass/HANCOCK_MultimodalDataset/features/targets.csv \
      --out_pkl     pid_index.pkl
"""

import argparse
import pandas as pd
from pathlib import Path

def main(targets_csv: str, out_pkl: str):
    targets_csv = Path(targets_csv).expanduser()
    out_pkl     = Path(out_pkl).expanduser()

    if not targets_csv.exists():
        raise FileNotFoundError(f"Cannot find {targets_csv}")

    pid = (
        pd.read_csv(targets_csv, usecols=['patient_id'])
          .drop_duplicates()
          .set_index('patient_id')
    )
    pid['row'] = range(len(pid))           # 0 … N-1 unique row IDs
    pid.to_pickle(out_pkl)

    print(f"✅  Saved {len(pid)} patient IDs → {out_pkl}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets_csv', required=True,
                    help='Path to features/targets.csv')
    ap.add_argument('--out_pkl', default='pid_index.pkl',
                    help='Destination .pkl file (default: pid_index.pkl)')
    args = ap.parse_args()
    main(args.targets_csv, args.out_pkl)
