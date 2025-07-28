#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_train_xattn.py
────────────────────
Cross-attention baseline for Hancock multimodal data.
Run in PyCharm: highlight the CONFIG block or the whole file (Shift+Alt+E).
"""

# ───────── CONFIG ───────────────────────────────────────────────
DATA_DIR = r"D:/sanket_experiments/modalities_in_numpy_version"      # folder with *.npy
SELECTED_MODALITIES = ["tabular" ] #, "text", "tma", "wsi"]
DROP_ROWS_MISSING_ALL = False
SEED   = 0
FOLDS  = 5
EPOCHS = 150
LR     = 3e-4
BATCH  = 64
D_MODEL = 512
RC_REC0_SURV1 = 0        # 0 = recurrence, 1 = survival
# ────────────────────────────────────────────────────────────────

import numpy as np, torch, random, matplotlib; matplotlib.use("Agg")
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
PLOT_DIR = Path("../plots"); PLOT_DIR.mkdir(exist_ok=True)

# ───────── helper: read modality → token & mask ────────────────
def load_tokens(mods, rc_col, drop_missing):
    modal_files = {
        "tabular": ("tabular_features.npy",              None),
        "tma"    : ("tma_features.npy",                  "tma_masks.npy"),
        "wsi"    : ("wsi_features_mean.npy",             "wsi_masks_mean.npy"),
        "text"   : ("text_features_4096.npy",            "text_masks.npy"),
    }

    y = np.load(f"{DATA_DIR}/labels_filtered_R1825_S730.npy")[:, rc_col]
    N = len(y)

    tokens, masks = [], []
    d_token = 4096                     # pad all to same width

    for m in mods:
        feat_file, mask_file = modal_files[m]
        X = np.load(f"{DATA_DIR}/{feat_file}").astype("float32")  # (N, d_m)
        if X.shape[1] < d_token:                      # right-pad
            X = np.pad(X, ((0,0),(0,d_token-X.shape[1])))

        if mask_file is None:
            msk = np.ones(N, dtype=np.uint8)
        else:
            msk = np.load(f"{DATA_DIR}/{mask_file}").any(1).astype(np.uint8)
            if drop_missing == False:                 # zero-fill missing rows
                X[msk == 0] = 0.0
        tokens.append(X)
        masks.append(msk[:,None])

    tok = np.stack(tokens, axis=1)      # (N, M, d_token)
    msk = np.concatenate(masks,1)       # (N, M)

    if drop_missing:
        keep = msk.any(1)
        tok, msk, y = tok[keep], msk[keep], y[keep]

    # simple NaN handling – fill with column medians
    col_median = np.nanmedian(tok, axis=0)
    idx_nan    = np.isnan(tok)
    tok[idx_nan] = np.take(col_median, np.where(idx_nan)[1])


    return tok, msk, y.astype("float32")

# ───────── Transformer encoder with modality mask ──────────────
class MMEncoder(torch.nn.Module):
    def __init__(self, d_in, d_model=D_MODEL, n_heads=8, n_layers=4):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model, n_heads, batch_first=True)
        self.enc  = torch.nn.TransformerEncoder(enc_layer, n_layers)
        self.cls  = torch.nn.Parameter(torch.randn(1,1,d_model))

    def forward(self, x, mask):                 # x (B,M,d_in)  mask (B,M)
        b = x.size(0)
        x = self.proj(x)
        cls = self.cls.expand(b,-1,-1)
        x = torch.cat([cls,x],1)                # prepend [CLS]
        pad = torch.cat([torch.zeros(b,1,device=x.device), 1-mask],1).bool()
        z  = self.enc(x, src_key_padding_mask=pad)
        return z[:,0]                           # CLS token

# ────────── one-fold train / eval ───────────────────────────────
def train_one_fold(tok_tr, msk_tr, y_tr, tok_va, msk_va, y_va,
                   epochs, batch, lr, fold_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fold {fold_id} – device:", device)

    enc = MMEncoder(tok_tr.shape[2]).to(device)
    head= torch.nn.Sequential(
            torch.nn.Linear(D_MODEL,128), torch.nn.ReLU(),
            torch.nn.Dropout(0.4), torch.nn.Linear(128,1)).to(device)
    opt  = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()),
                             lr=lr, weight_decay=1e-4)
    crit = torch.nn.BCEWithLogitsLoss()

    ds_tr = TensorDataset(
        torch.from_numpy(tok_tr), torch.from_numpy(msk_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(
        torch.from_numpy(tok_va), torch.from_numpy(msk_va), torch.from_numpy(y_va))
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch)

    losses, val_auc = [], []

    for ep in range(1, epochs+1):
        enc.train(); head.train();  tot=0
        for x, m, y in dl_tr:
            x,m,y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            logit = head(enc(x,m)).squeeze()
            loss  = crit(logit, y)
            loss.backward(); opt.step()
            tot += loss.item()*x.size(0)
        losses.append(tot/len(ds_tr))

        enc.eval(); head.eval(); preds=[]
        with torch.no_grad():
            for x,m,_ in dl_va:
                x,m = x.to(device), m.to(device)
                p = torch.sigmoid(head(enc(x,m))).cpu().numpy()
                preds.append(p)
        preds = np.concatenate(preds).ravel()
        auc   = roc_auc_score(y_va, preds)
        val_auc.append(auc)
        print(f"Epoch {ep:03}/{epochs}  loss={losses[-1]:.4f}  AUC={auc:.4f}")

    # plot
    fpr,tpr,_ = roc_curve(y_va,preds)
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(range(1,epochs+1),losses); ax[0].set_title("Train loss")
    ax[1].plot(fpr,tpr,label=f"AUC={auc:.3f}"); ax[1].plot([0,1],[0,1],'--')
    ax[1].legend(); plt.tight_layout()
    plt.savefig(PLOT_DIR/f"fold{fold_id}_ep{epochs}.png",dpi=250); plt.close()
    return auc

# ─────────── main CV loop ───────────────────────────────────────
tokens, masks, y = load_tokens(SELECTED_MODALITIES, RC_REC0_SURV1,
                               DROP_ROWS_MISSING_ALL)
print("Tokens", tokens.shape, "Mask", masks.shape, "y", y.shape)

cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
auc_scores=[]
for fold,(tr,va) in enumerate(cv.split(tokens, y),1):
    auc = train_one_fold(tokens[tr], masks[tr], y[tr],
                         tokens[va], masks[va], y[va],
                         epochs=EPOCHS, batch=BATCH, lr=LR, fold_id=fold)
    auc_scores.append(auc)
    print(f"Fold {fold} AUC={auc:.4f}")

print("\nMean AUC =", np.mean(auc_scores))
