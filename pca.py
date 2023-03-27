import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import random


def pca_encoding(path_to_data, encoding_dim=2, seed=0):

    df = pd.read_csv(path_to_data)
    x = df.drop(["y"], axis=1)
    x_centered = np.array(x - np.mean(x, axis=0))

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    pca = PCA(n_components=encoding_dim, svd_solver="full")
    pca.fit(x_centered)

    return pca.transform(x_centered)
