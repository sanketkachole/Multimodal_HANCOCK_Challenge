#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mean_pool_wsi_full.py
─────────────────────
Mean-pool every WSI slide and write one row per patient.

Produces a CSV with:
    • patient_id (keeps leading zeros)
    • HAS_PRIM_WSI, HAS_LN_WSI          ⟶ 0/1 flags
    • P_000 … P_255  (primary-tumour vector)
    • L_000 … L_255  (lymph-node vector)

USAGE
$ python mean_pool_wsi_full.py \
    --targets_csv  …/features/targets.csv \
    --out_csv      …/patient_wsi_vectors_mean.csv
"""
import argparse, re, glob, h5py, numpy as np, pandas as pd, os
from tqdm import tqdm

# ---------- edit these if your slide paths change ----------
PRIM_ROOT = '/N/project/Sanket_Slate_Project/Hancock_Dataset/WSI_UNI_encodings/WSI_PrimaryTumor'
LN_ROOT   = '/N/project/Sanket_Slate_Project/Hancock_Dataset/WSI_UNI_encodings/WSI_LymphNode'
PAT_PRIM  = re.compile(r'PrimaryTumor_HE_(\d+)\.h5$')
PAT_LN    = re.compile(r'LymphNode_HE_(\d+)\.h5$')
# -----------------------------------------------------------

H5_KEY      = 'features'
EMB_DIM     = 1024
MAX_PATCHES = 10_000           # 0 ⇒ use all patches

# ---------------------------------------------------------------------------

def mean_pool(h5_path: str) -> np.ndarray:
    """Return the mean of all patch embeddings in `h5_path`."""
    with h5py.File(h5_path, 'r') as h5:
        arr = h5[H5_KEY][:]
    if MAX_PATCHES and arr.shape[0] > MAX_PATCHES:
        sel = np.random.choice(arr.shape[0], MAX_PATCHES, replace=False)
        arr = arr[sel]
    return arr.mean(0).astype(np.float32)     # (1024,)

def build_slide_dict() -> dict[tuple[str, str], np.ndarray]:
    """Collect mean-pooled vectors for every slide on disk."""
    slide_vec: dict[tuple[str, str], np.ndarray] = {}

    # lymph-node slides
    for p in glob.glob(f'{LN_ROOT}/*.h5'):
        m = PAT_LN.search(p)
        if m:
            pid = m.group(1)                  # KEEP AS STRING
            slide_vec[(pid, 'L')] = mean_pool(p)

    # primary-tumour slides (may be in sub-folders)
    for p in glob.glob(f'{PRIM_ROOT}/**/*.h5', recursive=True):
        m = PAT_PRIM.search(p)
        if m:
            pid = m.group(1)
            slide_vec[(pid, 'P')] = mean_pool(p)

    return slide_vec

# ---------------------------------------------------------------------------

def main(targets_csv: str, out_csv: str) -> None:
    # Master patient list (guarantees one row per patient)
    targets = pd.read_csv(
        targets_csv,
        usecols=['patient_id'],
        dtype={'patient_id': str}
    )
    patient_ids = targets['patient_id'].unique().tolist()

    slide_vec = build_slide_dict()
    rows = []

    for pid in tqdm(patient_ids, desc='Writing CSV'):
        zp = slide_vec.get((pid, 'P'), np.zeros(EMB_DIM, np.float32))
        zl = slide_vec.get((pid, 'L'), np.zeros(EMB_DIM, np.float32))

        rec = {
            'patient_id'   : pid,
            'HAS_PRIM_WSI': int((pid, 'P') in slide_vec),
            'HAS_LN_WSI'  : int((pid, 'L') in slide_vec)
        }
        rec.update({f'P_{i:03d}': zp[i] for i in range(EMB_DIM)})
        rec.update({f'L_{i:03d}': zl[i] for i in range(EMB_DIM)})
        rows.append(rec)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'✅  Saved {len(rows)} patients → {out_csv}')

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets_csv', required=True,
                    help='CSV listing every patient_id (e.g. targets.csv)')
    ap.add_argument('--out_csv', required=True,
                    help='Destination CSV file')
    args = ap.parse_args()
    main(args.targets_csv, args.out_csv)
