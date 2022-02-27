import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.svm as svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def k_means(data):
    target=data.target
    features=data.data

    features = pd.DataFrame(data=data.data, columns=data.feature_names)
    #print(features)

    #Dados para comparacao
    target=pd.Series(data.target,name='target')
    #print(target)

    #PCA
    pca=PCA(2)
    reduzido=pd.DataFrame(pca.fit_transform(features),index=features.index,columns=['x','y'])
    
    #Aplica K-Means 
    num_clusters=2
    km=KMeans(n_clusters=num_clusters,init='k-means++',random_state=2)
    km.fit(reduzido)
    clusters=pd.Series(km.labels_,index=reduzido.index,name='Clusters')

    #Adiciona Resultados no df reduzido
    reduzido=pd.concat([clusters,target,reduzido],axis=1)
    centroids=km.cluster_centers_

    #Printa report
    print(
        f"==================================================\n"
        f"Classification report for classifier {km}:\n"
        f"{metrics.classification_report(target,clusters )}\n"
    )

    #Plot
    plt.title('K-Means Cancer')
    sns.scatterplot(data=reduzido, x='x', y='y', hue='Clusters', palette='tab10')
    #plt.legend('Beningno','maligno')
    #plt.scatter(reduzido['x'],reduzido['y'],c=reduzido['Clusters'],s=30)
    plt.scatter(centroids[:,0],centroids[:,1],c='r',marker='x',s=100)
    plt.savefig('results/cancer_kmeans'+str(num_clusters)+'.png')
    plt.show()

def SVM_class(data):
    target=data.target
    features=data.data

    #Divide os dados em treino e teste
    X_treino,X_teste,y_treino,y_teste=train_test_split(features,target,test_size=0.5)

    #SVM
    classifier=svm.SVC()
    classifier.fit(X_treino,y_treino)
    predicted=classifier.predict(X_teste)

    #Printa report
    print(
        f"==================================================\n"
        f"Classification report for classifier {classifier}:\n"
        f"{metrics.classification_report(y_teste, predicted)}\n"
    )

cancer=datasets.load_breast_cancer()
#print(cancer.DESCR)
k_means(cancer)
SVM_class(cancer)


