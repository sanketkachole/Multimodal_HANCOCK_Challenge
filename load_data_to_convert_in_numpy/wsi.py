###################################################
######### convert organiser's features from .csv to .npy #####################
###################################################
prim_csv = pd.read_csv("D:/sanket_experiments_clean/organiser_wsi_prim_patch_level_mean.csv").set_index('patient_id') # 1024 dimentions
lymp_csv = pd.read_csv("D:/sanket_experiments_clean/organiser_wsi_lymph_patch_level_mean.csv").set_index('patient_id')

all_pid = sorted(set(prim_csv.index) | set(lymp_csv.index))

X = np.zeros((len(all_pid), 1024), np.float32)  # dummy vector to store features from prim
Y = np.zeros((len(all_pid), 1024), np.float32)  # dummy vector to store features from lymph
M = np.zeros((len(all_pid), ), np.uint8)   # dummy vector to store Mask for prim
N = np.zeros((len(all_pid), ), np.uint8)   # dummy vector to store Mask for lymph


for i, pid in enumerate(all_pid):
    if pid in prim_csv.index:
        X[i] = prim_csv.loc[pid, [f'P_{j:03d}' for j in range(1024)]]
        M[i] = int(prim_csv.loc[pid, 'HAS_PRIM_WSI'])
    if pid in lymp_csv.index:
        Y[i] = lymp_csv.loc[pid , [f'L_{j:03d}' for j in range(1024)]]
        N[i] = int(lymp_csv.loc[pid, 'HAS_LN_WSI'])


np.save('organiser_wsi_primary_mean_features.npy',X)
np.save('organiser_wsi_primary_mask.npy', M)
np.save('organiser_wsi_lymph_mean_features.npy', Y)
np.save('organiser_wsi_lymph_maks.npy', N)


###################################################
######### convert uni2's features from .csv to .npy #####################
###################################################
prim_csv = pd.read_csv("D:/sanket_experiments_clean/shubham_uni_wsi_prim_patch_level_mean.csv").set_index('patient_id') # 1024 dimentions
lymp_csv = pd.read_csv("D:/sanket_experiments_clean/shubham_uni_wsi_lymph_patch_level_mean.csv").set_index('patient_id')

all_pid = sorted(set(prim_csv.index) | set(lymp_csv.index))

X = np.zeros((len(all_pid), 1024), np.float32)  # dummy vector to store features from prim
Y = np.zeros((len(all_pid), 1024), np.float32)  # dummy vector to store features from lymph
M = np.zeros((len(all_pid), ), np.uint8)   # dummy vector to store Mask for prim
N = np.zeros((len(all_pid), ), np.uint8)   # dummy vector to store Mask for lymph


for i, pid in enumerate(all_pid):
    if pid in prim_csv.index:
        X[i] = prim_csv.loc[pid, [f'P_{j:03d}' for j in range(1024)]]
        M[i] = int(prim_csv.loc[pid, 'HAS_PRIM_WSI'])
    if pid in lymp_csv.index:
        Y[i] = lymp_csv.loc[pid , [f'L_{j:03d}' for j in range(1024)]]
        N[i] = int(lymp_csv.loc[pid, 'HAS_LN_WSI'])


np.save('shubham_uni_wsi_prim_patch_level_mean_features.npy',X)
np.save('shubham_uni_wsi_prim_patch_level_mask.npy', M)
np.save('shubham_uni_wsi_lymph_patch_level_mean_features.npy', Y)
np.save('shubham_uni_wsi_lymph_patch_level_maks.npy', N)


###################################################
######### convert virchow's features from .csv to .npy #####################
###################################################
prim_csv = pd.read_csv("D:/sanket_experiments_clean/shubham_virchow2_wsi_prim_patch_level_mean.csv").set_index('patient_id') # 1024 dimentions
lymp_csv = pd.read_csv("D:/sanket_experiments_clean/shubham_virchow2_wsi_lymph_patch_level_mean.csv").set_index('patient_id')

all_pid = sorted(set(prim_csv.index) | set(lymp_csv.index))

X = np.zeros((len(all_pid), 1024), np.float32)  # dummy vector to store features from prim
Y = np.zeros((len(all_pid), 1024), np.float32)  # dummy vector to store features from lymph
M = np.zeros((len(all_pid), ), np.uint8)   # dummy vector to store Mask for prim
N = np.zeros((len(all_pid), ), np.uint8)   # dummy vector to store Mask for lymph


for i, pid in enumerate(all_pid):
    if pid in prim_csv.index:
        X[i] = prim_csv.loc[pid, [f'P_{j:03d}' for j in range(1024)]]
        M[i] = int(prim_csv.loc[pid, 'HAS_PRIM_WSI'])
    if pid in lymp_csv.index:
        Y[i] = lymp_csv.loc[pid , [f'L_{j:03d}' for j in range(1024)]]
        N[i] = int(lymp_csv.loc[pid, 'HAS_LN_WSI'])


np.save('shubham_virchow2_wsi_prim_patch_level_mean_features.npy',X)
np.save('shubham_virchow2_wsi_prim_patch_level_mask.npy', M)
np.save('shubham_virchow2_wsi_lymph_patch_level_mean_features.npy', Y)
np.save('shubham_virchow2_wsi_lymph_patch_level_maks.npy', N)




