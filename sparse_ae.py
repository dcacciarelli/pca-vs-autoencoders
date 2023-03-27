import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
import random
from tqdm import tqdm

# df = pd.read_csv("/Users/dcac/Data/Soft_Sensors/debutanizer.csv")
df = pd.read_csv("/Users/dcac/Data/Soft_Sensors/SRU1.csv")
X = df.drop(["y"], axis=1)
X_centered = np.array(X - np.mean(X, axis=0))
seed = 0
inp_shape = X.shape[1]


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_size, sparsity_weight):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoding_size, bias=False)
        self.decoder = nn.Linear(encoding_size, input_size, bias=False)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def custom_loss(self, compression, reconstruction, original):
        loss = torch.norm((reconstruction - original), p=2) + \
               self.sparsity_weight * torch.norm(compression, p=1)
        return loss

# Train the sparse autoencoder to reconstruct the input data with sparsity regularization
sparse_autoencoder = SparseAutoencoder(input_size=inp_shape, encoding_size=5, sparsity_weight=0.01)
optimizer = torch.optim.Adam(sparse_autoencoder.parameters(), lr=0.001)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
batch_size = 100
num_epochs = 1000
pbar = tqdm(total=num_epochs, desc="Training", position=0)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(X_centered), batch_size):
        batch = torch.tensor(X_centered[i:i + batch_size], dtype=torch.float32)
        optimizer.zero_grad()
        hidden = sparse_autoencoder.encoder(batch)
        recon = sparse_autoencoder.decoder(hidden)
        loss = sparse_autoencoder.custom_loss(hidden, recon, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / (len(X_centered) / batch_size)
    pbar.set_postfix(loss=f"{avg_loss:.6f}")
    pbar.update()

pbar.close()

# Encode the data using PCA and the autoencoder
encoded_data = sparse_autoencoder.encoder(torch.tensor(X_centered).float()).detach().numpy()

# Create a heatmap of the encoding matrix
sns.heatmap(encoded_data, cmap='viridis', cbar=True)
plt.xlabel('Encoding Dimension')
plt.ylabel('Input Sample')
plt.title('Encoding Matrix Heatmap')
plt.show()

# Plot a histogram for each dimension
fig, axs = plt.subplots(nrows=1, ncols=encoded_data.shape[1], figsize=(10, 4))
for i in range(encoded_data.shape[1]):
    axs[i].hist(encoded_data[:, i], bins=50)
    axs[i].set_title(f"Dimension {i+1}")
plt.tight_layout()
plt.show()

# Compute the covariance matrix of the encoded data
covariance_matrix = np.cov(encoded_data, rowvar=False)

# Plot the covariance matrix
plt.imshow(covariance_matrix)
plt.colorbar()
plt.show()