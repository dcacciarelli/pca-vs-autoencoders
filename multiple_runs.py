import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import os
import random

# Generate random data
inp_shape = 10
# num_runs = 10
seeds = np.arange(100) # different seeds for different runs

pca_weights_all = []
autoencoder_weights_all = []
pca_encoding_all = []
autoencoder_encoding_all = []

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    X = np.random.randn(1000, inp_shape)
    X_centered = X - np.mean(X, axis=0)

    # Perform PCA on the data
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(X_centered)
    pca_weights = pca.components_
    pca_weights_all.append(pca_weights)

    # Define a linear autoencoder with the same number of hidden units as PCA components
    autoencoder = Autoencoder(input_size=inp_shape, encoding_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.1)

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        recon = autoencoder(torch.tensor(X_centered, dtype=torch.float32))
        loss = criterion(recon, torch.tensor(X_centered).float())
        loss.backward()
        optimizer.step()

    # Extract the weights of the encoder
    autoencoder_weights = autoencoder.encoder.weight.detach().numpy()
    autoencoder_weights_all.append(autoencoder_weights)

    # Encode the data using PCA and the autoencoder
    pca_encoding = pca.transform(X_centered)
    pca_encoding_all.append(pca_encoding)
    autoencoder_encoding = autoencoder.encoder(torch.tensor(X_centered).float()).detach().numpy()
    autoencoder_encoding_all.append(autoencoder_encoding)

# Compute the average of the weights and encodings across the runs
pca_weights_avg = np.mean(pca_weights_all, axis=0)
autoencoder_weights_avg = np.mean(autoencoder_weights_all, axis=0)
pca_encoding_avg = np.mean(pca_encoding_all, axis=0)
autoencoder_encoding_avg = np.mean(autoencoder_encoding_all, axis=0)

# Plot the average weights of PCA and the autoencoder
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].bar(np.arange(inp_shape), pca_weights_avg[0])
axs[0, 0].set_title('PCA weights (component 1)')
axs[0, 1].bar(np.arange(inp_shape), autoencoder_weights_avg[0])
axs[0, 1].set_title('Autoencoder weights (component 1)')
axs[1, 0].bar(np.arange(inp_shape), pca_weights_avg[1])
axs[1, 0].set_title('PCA weights (component 2)')
axs[1, 1].bar(np.arange(inp_shape), autoencoder_weights_avg[1])
axs[1, 1].set_title('Autoencoder weights (component 2)')
plt.show()

# Plot the average encoded features using PCA and the autoencoder
fig, axs = plt.subplots(1, 2)
axs[0].scatter(pca_encoding_avg
