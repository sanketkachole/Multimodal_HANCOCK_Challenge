import pandas as pd, numpy as np
from pathlib import Path

CSV  = Path(r"Y:/HANCOCK_MultimodalDataset/features/tabular_features/targets.csv")
OUT  = "labels_R730_S1825.npy"          # 2-yr recurrence, 5-yr survival
REC_HOR, SURV_HOR = 730, 1825           # horizons in days

tgt = pd.read_csv(CSV)

# --- force numeric, turn blanks or bad strings into NaN -----------------
tgt["dtr"] = pd.to_numeric(tgt["days_to_recurrence"], errors="coerce")
tgt["dli"] = pd.to_numeric(tgt["days_to_last_information"], errors="coerce")

# ---------- derive both flags in one pass ---------------------
tgt = tgt.assign(
    recurrence_num = (
        (tgt["recurrence"].str.strip().str.lower() == "yes") &
        (tgt["dtr"].fillna(np.inf) <= REC_HOR)
    ).astype(np.uint8),
    survival_5y_num = (
        tgt["survival_status"].str.strip().str.lower().str.startswith("deceased") &
        (tgt["dli"].fillna(np.inf) <= SURV_HOR)
    ).astype(np.uint8)
)

# ---------- save label matrix ---------------------------------
y = tgt[["recurrence_num", "survival_5y_num"]].to_numpy(np.uint8)
np.save(OUT, y)
print("✅ saved", OUT, "shape", y.shape)

# quick sanity check: show rows where logic might differ
print(tgt.loc[tgt["recurrence_num"]==0 & (tgt["recurrence"].str.lower()=="yes") &
              (tgt["dtr"]>REC_HOR), ["patient_id","recurrence","dtr","recurrence_num"]].head())
