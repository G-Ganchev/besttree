from class_tree import *
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


titanic = pd.read_csv('titanic.txt')
X = titanic[['Pclass',  'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked']]
y = titanic['Survived']

X['Fare'] = X['Fare'].astype(float).fillna(X['Fare'].astype(float).mean())
X['Age'] = X['Age'].astype(float).fillna(X['Age'].astype(float).mean())
X = X.fillna('MISSING')

X_my = X.astype('category')
X_my['Fare'] = X_my['Fare'].astype(float)
X_my['Age'] = X_my['Age'].astype(float)
X_my.dtypes


X_sk = pd.get_dummies(X)



my_perf = []
sk_perf = []
for i in range(1,15):
    print(i)
    my_tree = BinaryClassificationTree(max_depth=i)
    my_tree.fit(X_my,y)
    pred = my_tree.predict(X_my)
    #print('accuracy my:' ,np.mean(pred == y))
    
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(X_sk,y)
    pred_sk  = clf.predict(X_sk)
    #print('accuracy sk:' ,np.mean(pred_sk == y))
    my_perf.extend([np.mean(pred == y)])
    sk_perf.extend([np.mean(pred_sk == y)])
    
plt.plot(range(1,15),my_perf,label= 'Categorical CART')
plt.plot(range(1,15),sk_perf,label='SKlearn CART ')
plt.title('SKlearn CART vs Categorical Cart - CONVERGENCE')
plt.xlabel('max_depth')
plt.ylabel('accurcay')
plt.legend()
