import argparse, re, glob, h5py, numpy as np, pandas as pd, os
from tqdm import tqdm

# ---------- edit these if your slide paths change ----------
PRIM_ROOT   = r"Y:/Hancock_Dataset/WSI_UNI_encodings/WSI_PrimaryTumor"
LN_ROOT     = r"Y:/Hancock_Dataset/WSI_UNI_encodings/WSI_LymphNode"
targets_csv = r"Y:/HANCOCK_MultimodalDataset/features/tabular_features/targets.csv"
out_csv_primary = r"D:/sanket_experiments_clean/organiser_wsi_prim_patch_level_mean.csv"
out_csv_lymph   = r"D:/sanket_experiments_clean/organiser_wsi_lymph_patch_level_mean.csv"

PAT_PRIM = re.compile(r'PrimaryTumor_HE_(\d+)\.h5$')
PAT_LN   = re.compile(r'LymphNode_HE_(\d+)\.h5$')
# -----------------------------------------------------------
H5_KEY      = 'features'
EMB_DIM     = 1024
MAX_PATCHES = 10_000           # 0 ⇒ use all patches

def mean_pool(h5_path: str) -> np.ndarray:
    """Return the mean of all patch embeddings in `h5_path`."""
    with h5py.File(h5_path, 'r') as h5:
        arr = h5[H5_KEY][:]
    if MAX_PATCHES and arr.shape[0] > MAX_PATCHES:
        sel = np.random.choice(arr.shape[0], MAX_PATCHES, replace=False)
        arr = arr[sel]
    if arr.shape[0] == 0:
        return np.zeros(EMB_DIM, np.float32)
    vec = arr.mean(0).astype(np.float32)
    # optional sanity check:
    assert vec.shape[0] == EMB_DIM, f"Expected {EMB_DIM}, got {vec.shape[0]}"
    return vec

def build_slide_dict() -> dict[tuple[str, str], np.ndarray]:
    """Collect mean-pooled vectors for every slide on disk."""
    slide_vec: dict[tuple[str, str], np.ndarray] = {}

    for p in glob.glob(f'{LN_ROOT}/*.h5'):
        m = PAT_LN.search(p)
        if m:
            pid = m.group(1)  # keep as string
            slide_vec[(pid, 'L')] = mean_pool(p)

    for p in glob.glob(f'{PRIM_ROOT}/**/*.h5', recursive=True):
        m = PAT_PRIM.search(p)
        if m:
            pid = m.group(1)
            slide_vec[(pid, 'P')] = mean_pool(p)

    return slide_vec

def write_two_csvs(targets_csv: str,
                   out_csv_primary: str,
                   out_csv_lymph: str,
                   drop_missing: bool = False) -> None:
    # ensure output folders exist
    os.makedirs(os.path.dirname(out_csv_primary), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv_lymph), exist_ok=True)

    # master patient list
    targets = pd.read_csv(targets_csv, usecols=['patient_id'], dtype={'patient_id': str})
    patient_ids = targets['patient_id'].unique().tolist()

    slide_vec = build_slide_dict()
    rows_P, rows_L = [], []

    for pid in tqdm(patient_ids, desc='Building primary / lymph CSVs'):
        # fetch vectors or zeros
        zp = slide_vec.get((pid, 'P'), np.zeros(EMB_DIM, np.float32))
        zl = slide_vec.get((pid, 'L'), np.zeros(EMB_DIM, np.float32))

        has_P = int((pid, 'P') in slide_vec)
        has_L = int((pid, 'L') in slide_vec)

        if (not drop_missing) or has_P:
            recP = {'patient_id': pid, 'HAS_PRIM_WSI': has_P}
            recP.update({f'P_{i:03d}': zp[i] for i in range(EMB_DIM)})
            rows_P.append(recP)

        if (not drop_missing) or has_L:
            recL = {'patient_id': pid, 'HAS_LN_WSI': has_L}
            recL.update({f'L_{i:03d}': zl[i] for i in range(EMB_DIM)})
            rows_L.append(recL)

    pd.DataFrame(rows_P).to_csv(out_csv_primary, index=False)
    pd.DataFrame(rows_L).to_csv(out_csv_lymph, index=False)
    print(f"✅ Saved {len(rows_P)} rows → {out_csv_primary}")
    print(f"✅ Saved {len(rows_L)} rows → {out_csv_lymph}")

# run
write_two_csvs(targets_csv, out_csv_primary, out_csv_lymph, drop_missing=False)




