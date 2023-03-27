import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoding_size, bias=False)
        self.decoder = nn.Linear(encoding_size, input_size, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def ae_encoding(x, encoding_dim=2, learning_rate=0.001, num_epochs=100, batch_size=32, optimizer=torch.optim.SGD):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    inp_shape = x.shape[1]
    autoencoder = Autoencoder(input_size=inp_shape, encoding_size=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optimizer(autoencoder.parameters(), lr=learning_rate)
    pbar = tqdm(total=num_epochs, desc="Training", position=0)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, len(x_scaled), batch_size):
            batch = torch.tensor(x_scaled[i:i + batch_size], dtype=torch.float32)
            optimizer.zero_grad()
            recon = autoencoder(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / (len(x_scaled) / batch_size)
        pbar.set_postfix(loss=f"{avg_loss:.6f}")
        pbar.update()
    pbar.close()

    return autoencoder.encoder(torch.tensor(x_scaled).float()).detach().numpy()
