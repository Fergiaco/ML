import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def kmeans(data):
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

    #Printa report
    print(
        f"Classification report for classifier {km}:\n"
        f"{metrics.classification_report(target,clusters)}\n"
        f"Precisao calculada = {precisao}:\n"
    )

    #Plot
    plt.title('K-Means Vinho')
    sns.scatterplot(data=reduzido, x='x', y='y', hue='Clusters', palette='tab10')
    #plt.scatter(reduzido['x'],reduzido['y'],c=reduzido['Clusters'],s=30)
    plt.scatter(centroids[:,0],centroids[:,1],c='r',marker='x',s=100,zorder=100,linewidths=3)
    plt.savefig('results/vinho_kmeans'+str(num_clusters)+'.png')
    plt.show()


def SVM_class(data):
    target=data.target
    features=data.data
    #print(features)
    #print(target)

    #Aplica escala nos dados
    scaler=StandardScaler()
    scaler.fit(features)
    featuresSc=scaler.transform(features)
    #print(featuresSc)

    #Divide os dados em treino e teste
    X_treino,X_teste,y_treino,y_teste=train_test_split(featuresSc,target,test_size=0.5)

    #SVM
    classifier=SVC()
    classifier.fit(X_treino,y_treino)
    predicted=classifier.predict(X_teste)

    #Printa report
    print(
        f"==================================================\n"
        f"Classification report for classifier {classifier}:\n"
        f"{metrics.classification_report(y_teste, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_teste, predicted)
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

#Carrega o dataset e converte pro Dataframe
data=load_wine()
#print(data.DESCR)

kmeans(data)
SVM_class(data)