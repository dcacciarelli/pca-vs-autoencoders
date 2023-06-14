import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_encoding(x, encoding_dim=2):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=encoding_dim, svd_solver="full")

    return pca.fit_transform(x_scaled), pca.components_
