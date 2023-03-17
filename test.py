import numpy as np
import torch
import os
import random
from sklearn.decomposition import PCA

# Generate random data
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
X = np.random.rand(100, 5)

# Center the data
X_centered = X - np.mean(X, axis=0)

# Perform SVD to obtain the top-2 principal components
U, s, V = np.linalg.svd(X_centered, full_matrices=False)
W = V[:2, :]


# Define the linear autoencoder model
class CustomAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, encoding_dim, W):
        super(CustomAutoencoder, self).__init__()
        self.encoder = torch.nn.Linear(input_dim, encoding_dim, bias=False)
        self.encoder.weight.data = torch.from_numpy(W.T).float()
        self.decoder = torch.nn.Linear(encoding_dim, input_dim, bias=False)
        self.decoder.weight.data = torch.from_numpy(W).float()

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


# Define the custom loss function
def custom_loss_function(x, y, W):
    x_projected = np.dot(x, W.T)
    y_projected = np.dot(y, W.T)
    reconstruction_error = np.mean(np.square(x_projected - y_projected))
    return reconstruction_error


# Train the linear autoencoder model
input_dim = X.shape[1]
encoding_dim = 2
model = CustomAutoencoder(input_dim, encoding_dim, W)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_tensor = torch.from_numpy(X_centered).float()
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = custom_loss_function(X_centered, output.numpy(), W)
    loss_tensor = torch.tensor(loss).float()
    loss_tensor.backward()
    optimizer.step()

# Get the encoded data using the linear autoencoder model
with torch.no_grad():
    encoded_tensor = model.encoder(X_tensor)
    Z_ae = encoded_tensor.numpy()

print("SVD result:\n", U[:, :2] @ np.diag(s[:2]))
print("Custom linear autoencoder result:\n", Z_ae)