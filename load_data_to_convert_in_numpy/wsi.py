#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_wsi_mean.py
────────────────
Convert *patient_wsi_vectors_mean.csv* → two NumPy arrays:

    • wsi_features_mean.npy   (N × 2048  float32)
    • wsi_masks_mean.npy      (N ×   2   uint8)   → [HAS_PRIM_WSI, HAS_LN_WSI]

CSV must have columns
    patient_id, HAS_PRIM_WSI, HAS_LN_WSI,
    P_000 … P_1023, L_000 … L_1023
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path

EMB_DIM = 1024                                   # ← 1024 per slide type
COL_P   = [f'P_{i:03d}' for i in range(EMB_DIM)]
COL_L   = [f'L_{i:03d}' for i in range(EMB_DIM)]

def main(wsi_csv, pid_pkl, out_feat, out_mask):
    # ---- patient index -----------------------------------------------------
    pid_df = pd.read_pickle(pid_pkl)              # index = patient_id
    n_pat  = len(pid_df)

    # row lookup: prefer an explicit "row" column; fall back to enumeration
    if 'row' in pid_df.columns:
        row_of = pid_df['row'].to_dict()
    else:
        row_of = {pid: i for i, pid in enumerate(pid_df.index)}

    # ---- read CSV ----------------------------------------------------------
    wsi_df = pd.read_csv(wsi_csv).set_index('patient_id')

    # ---- allocate ----------------------------------------------------------
    X = np.zeros((n_pat, EMB_DIM * 2), dtype=np.float32)
    M = np.zeros((n_pat, 2),             dtype=np.uint8)   # [prim, ln]

    # ---- fill --------------------------------------------------------------
    for pid, row in wsi_df.iterrows():
        if pid not in row_of:                       # safety check
            continue
        idx = row_of[pid]

        X[idx, :EMB_DIM]  = row[COL_P].to_numpy(dtype='float32')
        X[idx, EMB_DIM:]  = row[COL_L].to_numpy(dtype='float32')

        M[idx, 0]         = int(row['HAS_PRIM_WSI'])
        M[idx, 1]         = int(row['HAS_LN_WSI'])

    # ---- save --------------------------------------------------------------
    np.save(out_feat, X)
    np.save(out_mask, M)
    print(f"✅  Saved {out_feat}   shape={X.shape}")
    print(f"✅  Saved {out_mask}   shape={M.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi_csv',  required=True,
                    help='patient_wsi_vectors_mean.csv')
    ap.add_argument('--pid_pkl',  required=True,
                    help='pid_index.pkl')
    ap.add_argument('--out_feat', default='wsi_features_mean.npy',
                    help='Destination N×2048 float32 array')
    ap.add_argument('--out_mask', default='wsi_masks_mean.npy',
                    help='Destination N×2 uint8 presence flags')
    args = ap.parse_args()
    main(**vars(args))




