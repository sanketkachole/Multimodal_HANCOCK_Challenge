#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_tma.py
───────────
Create patient-aligned TMA feature matrix (N × 4096) plus presence masks.

USAGE
$ python load_tma.py \
    --npz_dir  /N/project/wsiclass/HANCOCK_MultimodalDataset/features \
    --pid_pkl  pid_index.pkl \
    --out_feat tma_features.npy \
    --out_mask tma_masks.npy
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path

STAINS = [
    "CD3", "CD8", "CD56", "CD68",
    "CD163", "HE", "MHC1", "PDL1"
]

def main(npz_dir, pid_pkl, out_feat, out_mask):
    npz_dir = Path(npz_dir)
    pid_df  = pd.read_pickle(pid_pkl)          # index = patient_id
    n_pat   = len(pid_df)

    X   = np.zeros((n_pat, 512 * len(STAINS)), dtype=np.float32)
    M   = np.zeros((n_pat, len(STAINS)),       dtype=np.uint8)   # mask per stain

    for i, stain in enumerate(STAINS):
        f = npz_dir / f"tma_tile_dtr_256_{stain}.npz"
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}")

        data = np.load(f, allow_pickle=False)
        # assume patient_id is the array key  (e.g. data['1'])
        for pid_str in data.files:
            pid = int(pid_str)
            if pid not in pid_df.index:
                continue
            row = pid_df.loc[pid, 'row']
            X[row, i*512:(i+1)*512] = data[pid_str]
            M[row, i] = 1

    np.save(out_feat, X)
    np.save(out_mask, M)
    print(f"✅  Saved {out_feat}  shape={X.shape}")
    print(f"✅  Saved {out_mask}  shape={M.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz_dir', required=True,
                    help='Folder containing the eight TMA .npz files')
    ap.add_argument('--pid_pkl', required=True,
                    help='pid_index.pkl')
    ap.add_argument('--out_feat', default='tma_features.npy',
                    help='Destination .npy for the 4096-D matrix')
    ap.add_argument('--out_mask', default='tma_masks.npy',
                    help='Destination .npy for the N×8 mask (0/1)')
    args = ap.parse_args()
    main(**vars(args))
