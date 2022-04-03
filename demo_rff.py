# Demo for testing Random Fourier Features module
import numpy as np
import sklearn
import rff.rfflearn.cpu as rfflearn                     # Import module

# Example 1: support vector classification with random matrix for CPU
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # Define input data
y = np.array([1, 1, 2, 2])                          # Defile label data
svc = rfflearn.RFFSVC().fit(X, y)                   # Training (on CPU)
score = svc.score(X, y)                             # Inference (on CPU)
print(score)
prediction = svc.predict(np.array([[-0.8, -1]]))
print(prediction)

# Example 2: Random Fourier Features with random matrix for CPU.
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])   # Define input data
dim_kernel = 128                                     # Dimension of Random Fourier Features
std_kernel = 0.1                                     # Standard deviation of Normal Random values
rff = rfflearn.RFF(rand_mat_type='rff', dim_kernel=dim_kernel, std_kernel=std_kernel)
rff = rff.fit(X)                                     # Random matrix computation for RFF (on CPU)
rffeat = rff.rff_compute(X)                          # RRF computation (on CPU)
print(rffeat[0][0, 0:5])
print(f'Shape data: {X.shape}')
print(f'Shape rff: {rffeat.shape}')

# Example 3: Random Projection Features with random matrix for CPU.
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])   # Define input data
dim_kernel = 128                                     # Dimension of Random Fourier Features
rff = rfflearn.RFF(rand_mat_type='rp', dim_kernel=dim_kernel, std_kernel=std_kernel)
rff = rff.fit(X)                                     # Random matrix computation for RFF (on CPU)
rpfeat = rff.rff_compute(X)                          # RRF computation (on CPU)
print(rpfeat[0][0:5])
print(f'Shape data: {X.shape}')
print(f'Shape rp: {rpfeat.shape}')