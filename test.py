from class_tree import *
import pandas as pd
import numpy as np



titanic = pd.read_csv('titanic.txt')
X = titanic[['Pclass',  'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked']]
y = titanic['Survived']

X['Fare'] = X['Fare'].astype(float).fillna(X['Fare'].astype(float).mean())
X['Age'] = X['Age'].astype(float).fillna(X['Age'].astype(float).mean())
X = X.fillna('MISSING')

X_cl = X.astype('category').copy()
X_cl['Fare'] = X_cl['Fare'].astype(float)
X_cl['Age'] = X_cl['Age'].astype(float)
X_cl.dtypes



tree = BinaryClassificationTree(max_depth=5)
tree.fit(X_cl,y)
pred = tree.predict(X_cl)
print(pred)