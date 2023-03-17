import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import os
import random
from tqdm import tqdm

df = pd.read_csv("/Users/dcac/Data/Soft_Sensors/debutanizer.csv")
X = df.drop(["y"], axis=1)
X_centered = np.array(X - np.mean(X, axis=0))
seed = 0
inp_shape = X.shape[1]

# Perform PCA on the data
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
pca = PCA(n_components=2, svd_solver="full")
pca.fit(X_centered)
pca_weights = pca.components_


# Define a linear autoencoder with the same number of hidden units as PCA components
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoding_size, bias=False)
        self.decoder = nn.Linear(encoding_size, input_size, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Train the autoencoder to reconstruct the input data
autoencoder = Autoencoder(input_size=inp_shape, encoding_size=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

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
        recon = autoencoder(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / (len(X_centered) / batch_size)
    pbar.set_postfix(loss=f"{avg_loss:.6f}")
    pbar.update()

pbar.close()

# Extract the weights of the encoder
autoencoder_weights = autoencoder.encoder.weight.detach().numpy()

# Plot the weights of PCA and the autoencoder
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].bar(np.arange(inp_shape), pca_weights[0])
axs[0, 0].set_title('PCA weights (component 1)')
axs[0, 1].bar(np.arange(inp_shape), autoencoder_weights[0])
axs[0, 1].set_title('Autoencoder weights (component 1)')
axs[1, 0].bar(np.arange(inp_shape), pca_weights[1])
axs[1, 0].set_title('PCA weights (component 2)')
axs[1, 1].bar(np.arange(inp_shape), autoencoder_weights[1])
axs[1, 1].set_title('Autoencoder weights (component 2)')
plt.show()

# Encode the data using PCA and the autoencoder
pca_encoding = pca.transform(X_centered)
autoencoder_encoding = autoencoder.encoder(torch.tensor(X_centered).float()).detach().numpy()

# Plot the encoded features using PCA and the autoencoder
fig, axs = plt.subplots(1, 2)
axs[0].scatter(pca_encoding[:, 0], pca_encoding[:, 1])
axs[0].set_title('PCA encoding')
axs[1].scatter(autoencoder_encoding[:, 0], autoencoder_encoding[:, 1])
axs[1].set_title('Autoencoder encoding')
plt.show()