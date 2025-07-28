
"""
from pathlib import Path
import glob, re, numpy as np, pandas as pd, torch

# ── CONFIG ────────────────────────────────────────────────
ROOT_DIR = Path(r"Y:/HANCOCK_MultimodalDataset/sanket_experiments/embeddings_output_4096")
PID_PKL  = Path(r"D:/sanket_experiments/pid_index.pkl") # these are simplely one coulmn with all the patient ID indexed from 0-762
OUT_FEAT = Path("text_features_12228.npy")
EMB_DIM  = 4096
# ─────────────────────────────────────────────────────────

# 1) Load patient index (index = patient_id, must have a 'row' column)
pid_df = pd.read_pickle(PID_PKL) # these are simplely one coulmn with all the patient ID indexed from 0-762
N_PAT   = len(pid_df)

# 2) Prepare the output matrix
X = np.zeros((N_PAT, EMB_DIM), dtype=np.float32)

# 3) Regex to extract patient ID from filenames like: patient_001.pt
pattern = re.compile(r"patient_(\d+)\.pt$")

# 4) Loop over all .pt files and fill X
for fn in glob.glob(str(ROOT_DIR / "*.pt")):
    m = pattern.search(Path(fn).name)
    if not m:                 # skip non-matching files
        continue
    pid = int(m.group(1))     # '001' → 1
    if pid not in pid_df.index:
        continue              # patient not in your index
    # .squeeze() Removes singleton dimensions (size 1).Example → a tensor of shape (1, 4096) becomes (4096,). Your files were saved as 1 × 4096, so this flattens them to a 1-D vector.
    # Converts the PyTorch tensor to a NumPy ndarray so the rest of the pipeline (NumPy matrix, saving with np.save, scikit-learn, etc.) can use it.
    vec = torch.load(fn, map_location="cpu").squeeze().numpy().astype("float32")
    X[pid_df.loc[pid, "row"]] = vec

# 5) Save (optional)
np.save(OUT_FEAT, X)
print(f"✅  Saved {OUT_FEAT}  shape={X.shape}")
"""
###########################################
###########################################
###########################################
###############################################
"""
from pathlib import Path
import re, numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(r"Y:/HANCOCK_MultimodalDataset/sanket_experiments/embeddings_output_4096")
PID_PKL  = Path(r"D:/sanket_experiments/pid_index.pkl")
OUT_FEAT = "text_features_12228.npy"
pattern  = re.compile(r"patient_(\d+)\.pt$")

# read patient index
pid_df = pd.read_pickle(PID_PKL)
X = np.zeros((len(pid_df), 4096), dtype=np.float32)

def loader(fn):
    pid = int(pattern.search(fn).group(1))
    if pid in pid_df.index:
        vec = torch.load(fn, map_location="cpu").squeeze().numpy()
        return pid, vec.astype("float32")
    return None

files = [str(p) for p in ROOT_DIR.glob("*.pt")]

with ThreadPoolExecutor(max_workers=8) as pool:     # tweak workers
    for item in pool.map(loader, files):
        if item is None:          # unknown patient
            continue
        pid, vec = item
        X[pid_df.loc[pid, "row"]] = vec

np.save(OUT_FEAT, X)
print("✅ saved", OUT_FEAT, X.shape)
"""
###########################################
########## with tqdm #####################
###########################################
###############################################

from tqdm import tqdm
import torch, numpy as np, pandas as pd, re
from pathlib import Path

ROOT_DIR = Path(r"Y:/HANCOCK_MultimodalDataset/sanket_experiments/embeddings_output_4096")
PID_PKL  = Path(r"D:/sanket_experiments/pid_index.pkl")
pattern  = re.compile(r"patient_(\d+)\.pt$")

pid_df = pd.read_pickle(PID_PKL)
X = np.zeros((len(pid_df), 4096), dtype=np.float32)

files = list(ROOT_DIR.glob("*.pt"))

for fn in tqdm(files, desc="Loading .pt files"):
    pid = int(pattern.search(fn.name).group(1))
    if pid not in pid_df.index:          # skip unknown IDs
        continue
    # .squeeze() Removes singleton dimensions (size 1).Example → a tensor of shape (1, 4096) becomes (4096,). Your files were saved as 1 × 4096, so this flattens them to a 1-D vector.
    # Converts the PyTorch tensor to a NumPy ndarray so the rest of the pipeline (NumPy matrix, saving with np.save, scikit-learn, etc.) can use it.
    vec = torch.load(fn, map_location="cpu").squeeze().numpy().astype("float32")
    X[pid_df.loc[pid, "row"]] = vec

np.save("text_features_4096.npy", X)                  # ← finally write to disk
print("✅ saved text_features_12228.npy", X.shape)

