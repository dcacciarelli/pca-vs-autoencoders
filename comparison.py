import matplotlib.pyplot as plt
import torch
from autoencoder import ae_encoding
from pca import pca_encoding

# Data paths
data = "/Users/dcac/Data/Soft_Sensors/debutanizer.csv"
# data = "/Users/dcac/Data/Soft_Sensors/SRU1.csv"
# data = "/Users/dcac/Data/UCI/air.csv"
# data = "/Users/dcac/Data/UCI/bike.csv"
# data = "/Users/dcac/Data/UCI/wine_white.csv"
# data = "/Users/dcac/Data/UCI/gas_turbine_co.csv"
# data = "/Users/dcac/Data/UCI/wine_red.csv"
# data = "/Users/dcac/Data/UCI/yacht.csv"
# data = "/Users/dcac/Data/UCI/housing.csv"

encoded_features_ae_sgd = ae_encoding(path_to_data=data, encoding_dim=2, learning_rate=0.01, num_epochs=1000, batch_size=32, optimizer=torch.optim.SGD, seed=0)
encoded_features_ae_adam = ae_encoding(path_to_data=data, encoding_dim=2, learning_rate=0.01, num_epochs=1000, batch_size=32, optimizer=torch.optim.Adam, seed=0)
encoded_features_pca = pca_encoding(path_to_data=data, encoding_dim=2, seed=0)

fig, axs = plt.subplots(1, 3, figsize=(12, 5))
axs[0].scatter(encoded_features_pca[:, 0], encoded_features_pca[:, 1], color="c", alpha=0.6)
axs[0].set_title('PCA encoding')
axs[1].scatter(encoded_features_ae_sgd[:, 0], encoded_features_ae_sgd[:, 1], color="slateblue", alpha=0.6)
axs[1].set_title('AE encoding (SGD)')
axs[2].scatter(encoded_features_ae_adam[:, 0], encoded_features_ae_adam[:, 1], color="crimson", alpha=0.6)
axs[2].set_title('AE encoding (Adam)')
plt.show()