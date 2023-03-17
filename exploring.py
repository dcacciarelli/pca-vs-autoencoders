import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate random data
X = np.random.rand(100, 5)

# Center the data
X_centered = X - np.mean(X, axis=0)

# Perform PCA
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(X_centered)

# Perform linear autoencoder using SVD
U, S, Vt = np.linalg.svd(X_centered)
E = Vt[:2, :].T  # Encoder matrix
D = Vt[:2, :2]   # Decoder matrix
Z_ae = X_centered.dot(E)

print("PCA result:\n", Z_pca)
print("Linear autoencoder result:\n", Z_ae)


# Visualize PCA result
plt.scatter(Z_pca[:, 0], Z_pca[:, 1])
plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Visualize linear autoencoder result
plt.scatter(Z_ae[:, 0], Z_ae[:, 1])
plt.title("Linear Autoencoder")
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.show()

