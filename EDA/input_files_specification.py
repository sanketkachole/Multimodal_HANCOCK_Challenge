#!/usr/bin/env python3
# quick_h5_to_xlsx.py
#
# Usage:  python quick_h5_to_xlsx.py  /path/file.h5
#
# Produces:  /path/file.xlsx   (two tabs: datasets, attributes)

import sys, json, h5py, pandas as pd, numpy as np
from pathlib import Path

h5_path = Path("/N/project/Sanket_Slate_Project/Hancock_Dataset/WSI_UNI_encodings/WSI_LymphNode/h5_files/LymphNode_HE_001.h5")
out_xlsx = Path("/N/project/wsiclass/HANCOCK_MultimodalDataset/lymphnode_h5_overview.xlsx").with_suffix(".xlsx")

dsets, attrs = [], []

with h5py.File(h5_path, "r") as h5:
    def see(name, obj):
        path = "/" + name if name else "/"
        if isinstance(obj, h5py.Dataset):         # ---- dataset row ----
            sh = obj.shape
            dsets.append({
                "path": path,
                "shape": sh,
                "dtype": str(obj.dtype),
                "preview": ",".join(f"{v:.4g}" for v in obj[0, :10])  # 1st vec, 10 cols
            })
        if obj.attrs:                             # ---- attribute row(s) ----
            for k, v in obj.attrs.items():
                # json-dump so lists/arrays show up nicely in Excel
                val = v.tolist() if isinstance(v, (np.ndarray, list, tuple)) else v
                attrs.append({"obj": path, "key": k, "value": json.dumps(val)})
    h5.visititems(see)

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xls:
    pd.DataFrame(dsets).to_excel(xls, "datasets", index=False)
    pd.DataFrame(attrs).to_excel(xls, "attributes", index=False)

print(f"wrote  {out_xlsx}")
