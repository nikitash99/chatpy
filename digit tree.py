import numpy as np
import pandas as pd
from sklearn import tree,naive_bayes
from sklearn.metrics import accuracy_score,confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sb

train_data = pd.read_csv('mnist_train.txt')
features_train = np.array(train_data.drop(['label'], 'columns'))
labels_train = np.array(train_data['label'])

test_data = pd.read_csv('mnist_test.txt')
features_test = np.array(test_data.drop(['label'], 'columns'))
labels_test = np.array(test_data['label'])

#clf=naive_bayes.GaussianNB()
#clf=neighbors.KNeighborsClassifier()  

clf = tree.DecisionTreeClassifier()

#training
t1=time.time()
clf=clf.fit(features_train, labels_train)
t2=time.time()
print("Training time:",t2-t1)

#testing
t3=time.time()
pre = clf.predict(features_test)
t4=time.time()
print("Testing time:",t4-t3)
print("Predicted:",pre)
print("Actual digit:",labels_test)

acc = accuracy_score(pre, labels_test)
print("Accuracy=",acc)

x=222
print("Predicted digit:",pre[x])
print("Actual digit:",labels_test[x])
digit=features_test[x]
digit_pixels=digit.reshape(28,28)
plt.imshow(digit_pixels,cmap="gray")
plt.show()

cm=confusion_matrix(labels_test,pre)
print("Confusion Matrix")
print(cm)

axis=plt.subplot()
sb.heatmap(cm, ax=axis, annot=True)
axis.set_xlabel("Predicted Digits")
axis.set_ylabel("Actual Digits")
axis.set_title("DecisionTree")
plt.show() 