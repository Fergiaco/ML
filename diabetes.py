import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, metrics, datasets,linear_model
from sklearn.model_selection import train_test_split

def linear_reg(data):
    
    target=data.target
    features=data.data
    #features=data.data[['age','bmi']]

    #Divide os dados em treino e teste
    X_treino,X_teste,y_treino,y_teste=train_test_split(features,target,test_size=0.5)

    reg=linear_model.LinearRegression()
    reg.fit(X_treino,y_treino)
    predicted=pd.Series(reg.predict(X_teste),index=X_teste.index,name='predicted')
    
    #print(pd.concat([y_teste,predicted],axis=1))

    #Printa report
    print('========================')
    print("Linear Regression")
    print('variance_score: %.2f' % metrics.explained_variance_score(y_teste, predicted))
    print("Mean squared error: %.2f" % metrics.mean_squared_error(y_teste, predicted))
    print("Coefficient of determination: %.2f" % metrics.r2_score(y_teste, predicted)) 
    #print("Coeficientes: ",reg.coef_)


def SVM_reg(data):

    target=data.target
    features=data.data
    #features=data.data[['age','bmi']]

    #Divide os dados em treino e teste
    X_treino,X_teste,y_treino,y_teste=train_test_split(features,target,test_size=0.5)

    #SVR    
    reg=svm.SVR()
    reg.fit(X_treino,y_treino)
    predicted=pd.Series(reg.predict(X_teste),index=X_teste.index,name='predicted')
    
    #print(pd.concat([predicted,y_teste],axis=1))
    
    #Printa report
    print('========================')
    print("SVM")
    print('variance_score: %.2f' % metrics.explained_variance_score(y_teste, predicted))
    print("Mean squared error: %.2f" % metrics.mean_squared_error(y_teste, predicted))
    print("Coefficient of determination: %.2f" % metrics.r2_score(y_teste, predicted)) 


#Carrega o dataset 
diabetes=datasets.load_diabetes(as_frame=True)

#print(diabetes.DESCR)
#  print(features.head())
#- age     age in years
#- sex
#- bmi     body mass index
#- bp      average blood pressure
#- s1      tc, total serum cholesterol
#- s2      ldl, low-density lipoproteins
#- s3      hdl, high-density lipoproteins
#- s4      tch, total cholesterol / HDL
#- s5      ltg, possibly log of serum triglycerides level
#- s6      glu, blood sugar level

linear_reg(diabetes)
SVM_reg(diabetes)
