import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate random data
X = np.random.rand(1000, 5)
X_torch = torch.from_numpy(X.astype(np.float32))

# Center the data
X_mean = torch.mean(X_torch, dim=0)
X_centered = X_torch - X_mean

# Perform PCA
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(X_centered.numpy())

# Define linear autoencoder model
class LinearAutoencoder(torch.nn.Module):
    def __init__(self, n_features, n_components):
        super(LinearAutoencoder, self).__init__()
        self.encoder = torch.nn.Linear(n_features, n_components, bias=False)
        self.decoder = torch.nn.Linear(n_components, n_features, bias=False)

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded

# Train linear autoencoder model
n_features = X.shape[1]
n_components = 2
model = LinearAutoencoder(n_features, n_components)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
n_epochs = 1000

for epoch in range(n_epochs):
    optimizer.zero_grad()
    X_reconstructed = model(X_torch)
    loss = criterion(X_reconstructed, X_torch)
    loss.backward()
    optimizer.step()

# Get encoded data from trained model
model.eval()
with torch.no_grad():
    Z_ae = model.encoder(X_torch).numpy()

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