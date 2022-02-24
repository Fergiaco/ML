import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

vinho=load_wine()
data = pd.DataFrame(data=vinho.data, columns=vinho.feature_names)
print(data)