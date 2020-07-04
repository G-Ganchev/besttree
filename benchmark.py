# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:03:48 2020

@author: GeorgiGanchev
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from class_tree import *

titanic = pd.read_csv('titanic.txt')
X = titanic[['Pclass',  'Sex', 'Age', 'SibSp',
        'Parch', 'Fare', 'Cabin', 'Embarked']]
y = titanic['Survived']

X['Fare'] = X['Fare'].astype(float).fillna(X['Fare'].astype(float).mean())
X['Age'] = X['Age'].astype(float).fillna(X['Age'].astype(float).mean())
X = X.fillna('MISSING')

X_cl = X.astype('category')
X_cl['Fare'] = X_cl['Fare'].astype(float)
X_cl['Age'] = X_cl['Age'].astype(float)
X_cl.dtypes


X_sk = pd.get_dummies(X)

cl_perf = []
c_py_perf = []
sk_perf = []
for i in range(1,15):
    print(i)
    cl_tree = BinaryClassificationTree(max_depth=i)
    cl_tree.fit(X_cl,y)
    pred = cl_tree.predict(X_cl)
    #print('accuracy cl:' ,np.mean(pred == y))
    
    clpy_tree = BinaryClassificationTreePy(max_depth=i)
    clpy_tree.fit(X_cl,y)
    predpy = clpy_tree.predict(X_cl)
    #print('accuracy cl:' ,np.mean(pred == y))
    
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(X_sk,y)
    pred_sk  = clf.predict(X_sk)
    #print('accuracy sk:' ,np.mean(pred_sk == y))
    cl_perf.extend([np.mean(pred == y)])
    c_py_perf.extend([np.mean(predpy == y)])
    sk_perf.extend([np.mean(pred_sk == y)])
    
plt.plot(range(1,15),cl_perf,label= 'Categorical CART (Cython)')
plt.plot(range(1,15),c_py_perf,label= 'Categorical CART (Python)')
plt.plot(range(1,15),sk_perf,label='SKlearn CART ')
plt.title('TITCANIC - SKlearn CART vs Categorical Cart - CONVERGENCE')
plt.xlabel('max_depth')
plt.ylabel('Accurcay')
plt.legend()
plt.show()



german_credit = pd.read_csv('german_credit.csv')

X = german_credit.drop(columns='default')
categorical_vars = list(X.dtypes[X.dtypes=='object'].index)
X_sk = pd.get_dummies(X[categorical_vars])

X_cl =X[categorical_vars].astype('category')

y = german_credit['default']




cl_perf = []
sk_perf = []
for i in range(1,20):
    print(i)
    cl_tree = BinaryClassificationTree(max_depth=i)
    cl_tree.fit(X_cl,y)
    #pred = cl_tree.predict(X_cl)
    pred = cl_tree.predict_proba(X_cl)
    #print('accuracy cl:' ,np.mean(pred == y))
    
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(X_sk,y)
    #pred_sk  = clf.predict(X_sk)
    pred_sk  = clf.predict_proba(X_sk)
    #print('accuracy sk:' ,np.mean(pred_sk == y))
    #cl_perf.extend([np.mean(pred == y)])
    #sk_perf.extend([np.mean(pred_sk == y)])
    cl_perf.extend([roc_auc_score(y,np.array(pred)[:,1]  )])
    sk_perf.extend([roc_auc_score(y, pred_sk[:,1] )])
    
    
plt.plot(range(1,20),cl_perf,label= 'Categorical CART')
plt.plot(range(1,20),sk_perf,label='SKlearn CART ')
plt.title('German Credit - SKlearn CART vs Categorical Cart - CONVERGENCE')
plt.xlabel('max_depth')
plt.ylabel('AUC')
plt.legend()
plt.show()
