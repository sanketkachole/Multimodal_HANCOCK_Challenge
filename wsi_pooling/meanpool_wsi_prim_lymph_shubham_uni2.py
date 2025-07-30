import argparse, re, glob, h5py, numpy as np, pandas as pd, os

from sympy import shape
from tqdm import tqdm

# ---------- edit these if your slide paths change ----------
ROOT   = r"Y:/HANCOCK_MultimodalDataset/features/shubham_wsi_primary_lymph_features/features/features_uni_v2"   # assign only one folder
targets_csv = r"Y:/HANCOCK_MultimodalDataset/features/tabular_features/targets.csv"
out_csv_primary = r"D:/sanket_experiments_clean/shubham_uni_wsi_prim_patch_level_mean.csv"
out_csv_lymph   = r"D:/sanket_experiments_clean/shubham_uni_wsi_lymph_patch_level_mean.csv"

PAT_BOTH = re.compile(r'^(PrimaryTumor|LymphNode)_HE_(\d+)\.h5$', re.IGNORECASE)

# -----------------------------------------------------------
H5_KEY      = 'features'
#EMB_DIM     = 1024
MAX_PATCHES = 10_000           # 0 ⇒ use all patches


def mean_pool(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, 'r') as h5:
        arr = h5[H5_KEY][:]
    if MAX_PATCHES and arr.shape[0] > MAX_PATCHES:
        sel = np.random.choice(arr.shape[0], MAX_PATCHES, replace=False)
        arr = arr[sel]
    if arr.shape[0] == 0:
        return np.zeros(arr.shape[1], np.float32)   # use file’s width
    return arr.mean(0).astype(np.float32)



def build_slide_dict() -> dict[tuple[str, str], np.ndarray]:
    slide_vec: dict[tuple[str, str], np.ndarray] = {}
    bad = []

    files = glob.glob(os.path.join(ROOT, "**", "*.h5"), recursive=True)
    for p in files:
        name = os.path.basename(p)
        m = PAT_BOTH.match(name)
        if not m:
            bad.append(name)
            continue

        kind_str, pid_raw = m.group(1), m.group(2)
        kind = 'P' if kind_str.lower().startswith('primary') else 'L'
        pid  = pid_raw.lstrip('0') or '0'   # normalize 000123 -> 123

        slide_vec[(pid, kind)] = mean_pool(p)

    print(f"Collected {len(slide_vec)} vectors; skipped {len(bad)} files")
    if bad:
        print("Unmatched examples:", bad[:10])
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
    targets['patient_id'] = targets['patient_id'].str.lstrip('0').replace({'': '0'})
    patient_ids = targets['patient_id'].unique().tolist()

    slide_vec = build_slide_dict()
    slide_vec = build_slide_dict()
    print("Vectors collected:", len(slide_vec))
    some_pid = next(iter(slide_vec))[0]
    print("Example PID from files:", some_pid)
    print("Is that PID in CSV?", some_pid in set(patient_ids))

    # detect dimensionality from the first available vector
    if len(slide_vec) == 0:
        raise RuntimeError("No slide vectors found under ROOT.")
    DIM = next(iter(slide_vec.values())).shape[0]

    rows_P, rows_L = [], []

    for pid in tqdm(patient_ids, desc='Building primary / lymph CSVs'):
        # fetch vectors or zeros
        zp = slide_vec.get((pid, 'P'), np.zeros(DIM, np.float32))
        zl = slide_vec.get((pid, 'L'), np.zeros(DIM, np.float32))

        has_P = int((pid, 'P') in slide_vec)
        has_L = int((pid, 'L') in slide_vec)

        if (not drop_missing) or has_P:
            recP = {'patient_id': pid, 'HAS_PRIM_WSI': has_P}
            recP.update({f'P_{i:03d}': zp[i] for i in range(DIM)})
            rows_P.append(recP)

        if (not drop_missing) or has_L:
            recL = {'patient_id': pid, 'HAS_LN_WSI': has_L}
            recL.update({f'L_{i:03d}': zl[i] for i in range(DIM)})
            rows_L.append(recL)

    pd.DataFrame(rows_P).to_csv(out_csv_primary, index=False)
    pd.DataFrame(rows_L).to_csv(out_csv_lymph, index=False)
    print(f"✅ Saved {len(rows_P)} rows → {out_csv_primary}")
    print(f"✅ Saved {len(rows_L)} rows → {out_csv_lymph}")

# run
write_two_csvs(targets_csv, out_csv_primary, out_csv_lymph, drop_missing=False)

