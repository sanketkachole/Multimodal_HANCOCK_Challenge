#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_train.py
──────────────
Minimal PyTorch baseline for Hancock multimodal data.
Edit the CONFIG block, highlight what you need, and run in PyCharm
(Shift + Alt + E).  No CLI, no hidden functions, no heavy OO.
"""

# ─────────────── CONFIG ──────────────────────────────────────────
DATA_DIR = r"D:/sanket_experiments_clean/modalities_in_numpy_version/"  # folder with *.npy
SELECTED_MODALITIES = ["tabular", "text", "wsi" , "tma"  ]#, "wsi" ]        # pick any subset ["tabular", "text", "wsi" , "tma"  ]
DROP_ROWS_MISSING_ALL = False                    # drop patients missing any selected modality?
SEED   = 0
FOLDS  = 5
EPOCHS_CFG = 150
LR     = 1e-4
BATCH  = 64
HIDDEN = 128
rcurnc_0_srvivl1 = 1    # put 0 for recurrence and 1 for survival
# ─────────────────────────────────────────────────────────────────

import os, json, numpy as np, torch, random
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
PLOT_DIR = Path("../plots"); PLOT_DIR.mkdir(exist_ok=True)

# reproducibility
torch.manual_seed(SEED);  np.random.seed(SEED);  random.seed(SEED)

# ─────────────── helper: load & fuse requested modalities ───────
def fuse_selected(modalities, rcurnc_0_srvivl1 = rcurnc_0_srvivl1, drop_missing=True):
    """
    modalities : list like ['tabular','text']
    Returns     : X (N, D_total) float32,  y (N,) uint8
    """
    # mapping from modality name → (feature file, mask file or None)
    modal_files = {
        "tabular": ("tabular_features.npy",              None),
        "tma"    : ("tma_features.npy",                  "tma_masks.npy"),
        "wsi"    : ("wsi_features_mean.npy",             "wsi_masks_mean.npy"),
        "text"   : ("text_features_4096.npy",                 "text_masks.npy"),
    }

    y = np.load(Path(DATA_DIR)/"labels_filtered_R1825_S730.npy")[:,rcurnc_0_srvivl1]    # (N,) # column 0 has recurrence and column 1 has survival labels ie. independent variable for the model
    N = len(y)

    keep = np.ones(N, dtype=bool)
    X_parts = []

    for m in modalities:
        feat_file, mask_file = modal_files[m]
        X = np.load(Path(DATA_DIR)/feat_file).astype("float32")
        X_parts.append(X)

        if mask_file is not None and drop_missing:
            mask = np.load(Path(DATA_DIR)/mask_file)     # (N, ?)
            keep &= mask.any(1)

    X_fused = np.concatenate(X_parts, axis=1)
    if drop_missing:
        X_fused, y = X_fused[keep], y[keep]

    # simple NaN handling – fill with column medians
    col_median = np.nanmedian(X_fused, axis=0)
    idx_nan    = np.isnan(X_fused)
    X_fused[idx_nan] = np.take(col_median, np.where(idx_nan)[1])

    return X_fused, y




# ─────────────── helper: one-fold PyTorch train / eval ───────────
def train_one_fold(X_train, y_train, X_val, y_val, hidden_sizes, dropout, epochs, batch_size, lr, fold_id):

    # ----- GPU/CPU device detection -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    D = X_train.shape[1]  # D is the input dimension (number of features after fusing selected modalities

    # ----- build a deeper MLP with dropout -----
    layers = []
    in_dim = D
    for h in hidden_sizes:
        layers += [
            torch.nn.Linear(in_dim, h), # Linear(in_dim, h) → a fully-connected layer mapping input → hidden dimension.
            torch.nn.ReLU(),            # ReLU() activation
            torch.nn.Dropout(dropout)
        ]
        in_dim = h
    layers += [torch.nn.Linear(in_dim, 1)]  # final logit
    net = torch.nn.Sequential(*layers).to(device)



    # ----- optimizer + loss -----
    opt  = torch.optim.Adam(net.parameters(), lr=lr)   # Adam with a fixed learning rate LR
    crit = torch.nn.BCEWithLogitsLoss()                # Loss: BCEWithLogitsLoss, which combines: a final sigmoid activation, binary cross-entropy in one numerically stable function. So the network output is a logit, and the loss expects a float label (0.0 or 1.0).

    # ----- wrap datasets -----
    ds_tr = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype("float32"))) # Converts NumPy arrays into PyTorch tensors. Wraps them in TensorDataset (pairs features & labels).
    ds_va = TensorDataset(torch.from_numpy(X_val  ), torch.from_numpy(y_val.astype("float32")))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True) # Wraps dataset in DataLoader: dl_tr will shuffle each epoch.
    dl_va = DataLoader(ds_va, batch_size=batch_size)  # dl_va just batches in order.


    # ----- tracking -----
    train_losses = []
    val_aucs = []


    for epoch in range(1, epochs+1): # Epoch loop runs for EPOCHS
        net.train()         # net.train() switches layers like dropout/batchnorm (not present here) into training mode.
        epoch_loss = 0.0

        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(net(xb).squeeze(), yb) # Forward pass: net(xb) → logits. squeeze() removes the last dim (shape [batch] not [batch,1]).
            loss.backward()  # Backward pass: loss.backward() computes gradients.
            opt.step()   # opt.step() updates weights.
            epoch_loss += loss.item() * xb.size(0)

        # mean loss
        epoch_loss /= len(dl_tr.dataset)
        train_losses.append(epoch_loss)


        net.eval()  # net.eval() disables training-specific layers (none here).
        preds = []
        with torch.no_grad():    # torch.no_grad() disables gradient tracking for inference.
            for xb,_ in dl_va:
                xb = xb.to(device)
                logits = net(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)

        preds = np.concatenate(preds).ravel()  # Loops over validation batches → collects sigmoid probabilities.
        auc = roc_auc_score(y_val, preds) # Computes AUC on validation set and returns it.
        val_aucs.append(auc)

        print(f"Epoch {epoch:02d}/{epochs} | train_loss={epoch_loss:.4f} | val_AUC={auc:.4f}")


    # ----- final eval predictions for ROC curve -----
    fpr, tpr, _ = roc_curve(y_val, preds)

    # ----- plot training curves -----
    fig, ax = plt.subplots(1, 2, figsize=(10,4))

    # 1)  # loss over epochs
    ax[0].plot(range(1, epochs+1), train_losses, marker='o')
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("BCE Loss")

    # 2) ROC curve
    ax[1].plot(fpr, tpr, label=f"AUC={val_aucs[-1]:.3f}")
    ax[1].plot([0,1],[0,1],'--',color='gray')
    ax[1].set_title("Validation ROC Curve")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend()

    plt.tight_layout()
    fname = PLOT_DIR / f"fold{fold_id}_ep{epochs}.png"
    plt.savefig(fname, dpi=250)
    plt.close()
    #plt.show()

    return val_aucs[-1], train_losses, val_aucs


# ─────────────── main block you can step through ────────────────
X, y = fuse_selected(SELECTED_MODALITIES, drop_missing=DROP_ROWS_MISSING_ALL)
print("Fused data :", X.shape,  "Labels:", y.shape)

cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
scores = []

for fold,(tr,va) in enumerate(cv.split(X,y), 1):
    fold_auc, train_losses, val_aucs = train_one_fold(
        X[tr], y[tr], X[va], y[va],
        hidden_sizes=[256,128,64],  # deeper MLP
        dropout=0.4,
        epochs=EPOCHS_CFG,
        batch_size=BATCH,
        lr=LR,
        fold_id=fold
    )
    scores.append(fold_auc)
    print(f"Fold {fold}/{FOLDS}  AUC={fold_auc:.4f}")

print(f"\nMean AUC across folds = {np.mean(scores):.4f}")

# → Ready to swap in any PyTorch model.
#   e.g. replace the `net = ...` block with a ResNet, TabNet, transformer, etc.



