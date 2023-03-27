import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
import random
from autoencoder import ae_encoding
from pca import pca_encoding
from sklearn import datasets

# Datasets
df = pd.read_csv("/Users/dcac/Data/Soft_Sensors/debutanizer.csv")
# df = pd.read_csv("/Users/dcac/Data/Soft_Sensors/SRU1.csv")
# df = pd.read_csv("/Users/dcac/Data/UCI/air.csv")
# df = pd.read_csv("/Users/dcac/Data/UCI/bike.csv")
# df = pd.read_csv("/Users/dcac/Data/UCI/wine_white.csv")
# df = pd.read_csv("/Users/dcac/Data/UCI/gas_turbine_co.csv")
# df = pd.read_csv("/Users/dcac/Data/UCI/power1.csv")
# data = "/Users/dcac/Data/UCI/wine_red.csv"
# data = "/Users/dcac/Data/UCI/yacht.csv"
features, y = datasets.load_iris().data, datasets.load_iris().target

# Splitting into X and y
features = df.drop(["y"], axis=1)
y = df["y"]

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
encoded_features_ae_sgd = ae_encoding(x=features, encoding_dim=2, learning_rate=0.01, num_epochs=1000, batch_size=32, optimizer=torch.optim.SGD)
encoded_features_ae_adam = ae_encoding(x=features, encoding_dim=2, learning_rate=0.01, num_epochs=1000, batch_size=32, optimizer=torch.optim.Adam)
encoded_features_pca = pca_encoding(x=features, encoding_dim=2)


fig, axs = plt.subplots(1, 3, figsize=(12, 5))
pca_plot = axs[0].scatter(encoded_features_pca[:, 0], encoded_features_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
axs[0].set_title('PCA encoding')
sgd_plot = axs[1].scatter(encoded_features_ae_sgd[:, 0], encoded_features_ae_sgd[:, 1], c=y, cmap='viridis', alpha=0.6)
axs[1].set_title('AE encoding (SGD)')
adam_plot = axs[2].scatter(encoded_features_ae_adam[:, 0], encoded_features_ae_adam[:, 1], c=y, cmap='viridis', alpha=0.6)
axs[2].set_title('AE encoding (Adam)')
cbar = fig.colorbar(adam_plot)
cbar.ax.set_ylabel('y label')
plt.show()
