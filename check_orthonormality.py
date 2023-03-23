import numpy as np

# Generate random data matrix X
X = np.random.rand(5, 10)

# Compute covariance matrix S
S = np.cov(X)

# Perform eigenvalue decomposition on S
V, C = np.linalg.eig(S)

# Sort eigenvalues in decreasing order and corresponding eigenvectors
idx = np.argsort(V)[::-1]
V = V[idx]

# Select top k eigenvectors to obtain V_k and Lambda_k
k = 3
V_k = V[:k]
C_k = C[: k, :]
Lambda_k = np.diag(V_k)
Lambda = np.diag(V)

# Verify orthonormality of C
print(np.round(C @ C_k.T, decimals=5))
print(np.round(C_k @ C.T, decimals=5))

# Verify diagonal matrix Lambda_k
print(np.round(Lambda_k, decimals=5))
print(np.round(C_k @ C.T @ Lambda @ C @ C_k.T, decimals=5))
