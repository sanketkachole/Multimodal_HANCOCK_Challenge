#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuse_modalities.py
──────────────────
Fuse pre-extracted modality feature *.npy files into one master feature
matrix (X), a per-modality presence mask, and the labels.  A 'slice_map'
dict is embedded in the .npz so downstream code can grab column ranges
*without any hard-wired numbers*.

Required files (same row-count N in each)
──────────────────────────────────────────
labels.npy
tabular_features.npy
tma_features.npy        + tma_masks.npy
wsi_features_mean.npy   + wsi_masks_mean.npy
text_features_12228.npy       + text_masks.npy

Output
──────────────────────────────────────────
patient_feature_matrix.npz
    • X          (N, D_total)   float32
    • mask       (N, 4)         uint8      (tab, tma, wsi, text)
    • y          (N, 2)         uint8
    • slice_map  object         dict {modality → (start, end)}
"""
import argparse, sys, numpy as np
from pathlib import Path

# ───────────────────────── helpers ───────────────────────────────
def load_npy(fname, desc):
    p = Path(fname)
    if not p.exists():
        sys.exit(f"❌  File not found: {fname}")
    arr = np.load(p, allow_pickle=False)
    print(f"· loaded {desc:<8} {arr.shape}")
    return arr

# ───────────────────────── main ──────────────────────────────────
def main(out_npz: str):
    # -------- modality → (feature file, mask file or None) -------
    modal_files = [
        ('tabular', 'tabular_features.npy',      None),
        ('tma',     'tma_features.npy',          'tma_masks.npy'),
        ('wsi',     'wsi_features_mean.npy',     'wsi_masks_mean.npy'),
        ('text',    'text_features_12228.npy',         'text_masks.npy'),
    ]

    y = load_npy('labels.npy', 'labels')
    N = y.shape[0]

    X_parts   = []
    mask_cols = []
    slice_map = {}
    offset    = 0

    for name, f_feat, f_mask in modal_files:
        Xmod = load_npy(f_feat, f"{name}-X")
        if Xmod.shape[0] != N:
            sys.exit(f"❌  Row mismatch in {f_feat}: {Xmod.shape[0]} vs {N}")

        # ----- remember column span for this modality -------------
        slice_map[name] = (offset, offset + Xmod.shape[1])
        offset += Xmod.shape[1]
        X_parts.append(Xmod)

        # ----- build presence mask column -------------------------
        if f_mask is None:        # tabular: always present
            mask_cols.append(np.ones((N,1), dtype=np.uint8))
        else:
            m = load_npy(f_mask, f"{name}-mask")
            if m.shape[0] != N:
                sys.exit(f"❌  Row mismatch in {f_mask}: {m.shape[0]} vs {N}")
            mask_cols.append((m.any(1, keepdims=True)).astype(np.uint8))

    # -------- fuse & save ----------------------------------------
    X    = np.concatenate(X_parts,   axis=1).astype('float32')
    mask = np.concatenate(mask_cols, axis=1)        # uint8
    np.savez(
        out_npz,
        X=X,
        mask=mask,
        y=y,
        slice_map=np.array([slice_map], dtype=object)  # dict in object array
    )

    print(f"\n✅  Saved {out_npz}")
    print(f"   X         {X.shape}  float32")
    print(f"   mask      {mask.shape}  uint8")
    print(f"   y         {y.shape}  {y.dtype}")
    print(f"   slice_map {slice_map}")

# ───────────────────────── CLI entry -point ──────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", default="patient_feature_matrix.npz",
                    help="Destination .npz (default: patient_feature_matrix.npz)")
    main(**vars(ap.parse_args()))
