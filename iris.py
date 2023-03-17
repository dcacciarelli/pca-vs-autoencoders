from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import os
import seaborn as sns
import random


iris = datasets.load_iris()
x = iris.data
y = iris.target
names = iris.target_names
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
pca = PCA(n_components=2, svd_solver="full")
pca.fit(x_scaled)
df = pd.DataFrame(pca.transform(x_scaled), columns=["PC1", "PC2"])
df["y"] = y
plt.figure(figsize=(10, 10))
sns.lmplot(x="PC1", y="PC2", data=df, fit_reg=False, hue='y', legend=False, palette="Set2")
plt.legend(loc='lower right')
plt.show()

