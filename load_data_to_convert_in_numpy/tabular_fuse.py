#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_tabular.py
───────────────
Assemble the six tabular CSVs **plus the targets file** into two NumPy arrays
aligned to *pid_index.pkl*:
    •  `tabular_features.npy`  – float32  (N × K)
    •  `labels.npy`            – int64    (N × 2) → [recurrence (0/1), surv5yr (0/1)]

Label logic
===========
* **Recurrence‑risk**  → 1 if `recurrence == "yes"`, else 0.  (Columns
  `rfs_event` and `days_to_rfs_event` already agree, but we use the text flag for
  clarity.)
* **Five‑year survival** (binary)  → 1 = **died within 5 years**  (i.e.\
  `survival_status == "deceased"` *and* `days_to_last_information < 1825`); 0
  otherwise (alive ≥5 years, or censored ≥5 years).

USAGE
-----
```bash
python load_tabular.py \
       --csv_dir  /N/project/wsiclass/HANCOCK_MultimodalDataset/features \
       --pid_pkl  pid_index.pkl \
       --out_feat tabular_features.npy \
       --out_lab  labels.npy
```
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

CSV_FILES = [
    "blood.csv",
    "clinical.csv",
    "icd_codes.csv",
    "pathological.csv",
    "tma_cell_density.csv",
]
TARGET_FILE = "targets.csv"


def main(csv_dir: str, pid_pkl: str, out_feat: str, out_lab: str):
    csv_dir = Path(csv_dir)

    # ---------------- master patient index ----------------
    pid_df = pd.read_pickle(pid_pkl)  # index = patient_id
    n_pat  = len(pid_df)

    # ---------------- aggregate tabular features ----------------
    feat_frames = []
    for fname in CSV_FILES:
        f = csv_dir / fname
        if not f.exists():
            raise FileNotFoundError(f"❌ Missing {f}")
        df = pd.read_csv(f).set_index("patient_id")
        feat_frames.append(df)
    features = pd.concat(feat_frames, axis=1)

    # ---------------- build label matrix ----------------
    tgt = pd.read_csv(csv_dir / TARGET_FILE).set_index("patient_id")

    #  Recurrence label: yes → 1, no / NaN → 0
    y_recur = (tgt["recurrence"].str.lower() == "yes").astype(np.int64)

    #  5‑year survival label: died < 5 yrs → 1, else 0
    died   = tgt["survival_status"].str.lower() == "deceased"
    lt5yr  = tgt["days_to_last_information"].fillna(99999) < 1825
    y_surv = (died & lt5yr).astype(np.int64)

    labels = pd.DataFrame({"recurrence": y_recur, "survival_5yr": y_surv})

    # ---------------- align to patient index ----------------
    features = features.reindex(pid_df.index)
    labels   = labels.reindex(pid_df.index)

    assert len(features) == n_pat, "Row mismatch after reindex (features)"
    assert len(labels)   == n_pat, "Row mismatch after reindex (labels)"

    # ---------------- convert & save ----------------
    X = features.to_numpy(dtype=np.float32)
    y = labels.to_numpy(dtype=np.int64)

    np.save(out_feat, X)
    np.save(out_lab, y)

    print(f"✅  Saved {out_feat}  shape={X.shape}")
    print(f"✅  Saved {out_lab}   shape={y.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True,
                    help="Path to folder containing the 6 tabular CSVs + targets.csv")
    ap.add_argument("--pid_pkl", required=True,
                    help="pid_index.pkl created earlier")
    ap.add_argument("--out_feat", default="tabular_features.npy",
                    help="Destination .npy for feature matrix")
    ap.add_argument("--out_lab",  default="labels.npy",
                    help="Destination .npy for label matrix")
    args = ap.parse_args()
    main(**vars(args))
