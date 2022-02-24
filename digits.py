import matplotlib.pyplot as plt
from sklearn import svm, metrics, datasets
from sklearn.model_selection import train_test_split

#Carrega o dataset 
digits=datasets.load_digits()
#print(digits.data)

#Printa imagens com label
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("T: %i" % label)
plt.show()

num_samples=len(digits.images)
data = digits.images.reshape((num_samples, -1))
#print(data[0])

#Divide os dados em treino e teste
X_treino,X_teste,y_treino,y_teste=train_test_split(data,digits.target,test_size=0.5)

#SVM    #gamma=0.001
classifier=svm.SVC()
classifier.fit(X_treino,y_treino)
predicted=classifier.predict(X_teste)

#Plot Imagens + Resultado
_, axes = plt.subplots(nrows=1, ncols=15, figsize=(15, 3))
for ax, image, prediction in zip(axes, X_teste, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"P: {prediction}")
plt.show()

#Printa report
print(
    f"Classification report for classifier {classifier}:\n"
    f"{metrics.classification_report(y_teste, predicted)}\n"
)

#Matriz de confusao
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_teste, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()