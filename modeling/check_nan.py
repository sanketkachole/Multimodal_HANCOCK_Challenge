import numpy as np
d   = np.load('../modalities_in_numpy_version/patient_feature_matrix.npz')
X   = d['X']
print("total NaNs:", np.isnan(X).sum())