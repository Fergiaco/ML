
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Carrega o dataset e converte pro Dataframe
data=load_wine(as_frame=True)
#print(data.DESCR)

#Dados para analise
features = pd.DataFrame(data=data.data, columns=data.feature_names)
#print(features.head())

#Dados para comparacao
target=pd.Series(data.target,name='target')
#print(target)

#Aplica escala nos dados
scaler=StandardScaler()
scaler.fit(features)
featuresSc=pd.DataFrame(scaler.transform(features),index=features.index,columns=features.columns)
#print(featuresSc.head())

#PCA
pca=PCA(2)
reduzido=pd.DataFrame(pca.fit_transform(featuresSc),index=features.index,columns=['x','y'])
#print(reduzido.head())

#Aplica K-Means 
num_clusters=3
km=KMeans(n_clusters=num_clusters,init='k-means++',random_state=10)
km.fit(featuresSc)
clusters=pd.Series(km.labels_,index=reduzido.index,name='Clusters')

#Adiciona Resultados no df reduzido
reduzido=pd.concat([clusters,target,reduzido],axis=1)
centroids=pca.transform(km.cluster_centers_)

#Calcula Precisao 
total=0
cont=0
for x in reduzido.values:
    x=x[:2]
    total+=1
    if x[0]==x[1]:
        cont+=1

precisao=cont/total*100
print('Precisao=',precisao)

#Plot
#sns.relplot(data=reduzido, x='x', y='y', hue='Clusters', palette='tab10', kind='scatter')
plt.title('K-Means Vinho')
plt.scatter(reduzido['x'],reduzido['y'],c=reduzido['Clusters'],s=30)
plt.scatter(centroids[:,0],centroids[:,1],c='r',marker='x',s=100)
plt.savefig('results/kmeans'+str(num_clusters)+'.png')
plt.show()



