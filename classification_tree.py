# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:06:06 2020

@author: GeorgiGanchev
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:56:46 2020

@author: GeorgiGanchev
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_object_dtype
from pandas.api.types import is_string_dtype



import numpy as np


class Node:
    def __init__(self, predicted_class,probability,node_type='leaf'):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.node_type = node_type
        self.probability = probability


class BinaryClassificationTreePy:
    def __init__(self, max_depth=float("inf")):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        X,y = self._prep_data(X,y)
        #print(X[0:5,:])
        self.tree_ = self._grow_tree(X, y)
        

    def predict(self, X):
        X = X[self.feature_names]
        X = X.values
        return [self._predict(inputs) for inputs in X]
    def predict_proba(self, X):
        X = X[self.feature_names]
        X = X.values
        return [self._predict(inputs,ptype='prob') for inputs in X]
    
    
    def _prep_data(self,X,y):
        self.feature_names = X.columns
        self.classes = np.unique(y)
        self.numeric_features = self._get_dtype(X,'numeric')
        self.string_features = self._get_dtype(X,'string')
        self.categorical_features = self._get_dtype(X,'categorical')
        self.date_features = self._get_dtype(X,'date')
        self.bool_features = self._get_dtype(X,'bool')
        self.feature_map = dict(zip(self.feature_names ,range(len(self.feature_names ))))
        self.rev_feature_map = dict(zip(self.feature_map.values(),self.feature_map.keys()))
        self.taget_map = dict(zip(self.classes ,range(len(self.classes ))))
        self.rev_taget_map = dict(zip(self.taget_map.values(),self.taget_map.keys()))
        self.feature_value_maps = {}
        return X.values,y.map(self.taget_map )
        

    def _get_dtype(self,X,d_type):
        dtype_func = {'numeric':is_numeric_dtype,
                      'string':is_string_dtype,
                      'categorical':is_categorical_dtype,
                      'date':is_datetime64_any_dtype,
                      'bool':is_bool_dtype}
        cols = []
        for f in X.columns:
            if dtype_func[d_type](X[f]):
                cols.extend([f])
        return cols
    def _format_categorical(self,X,y,idx):
        df = pd.DataFrame(X[:,idx],columns = [idx])
        df['y'] = y
        df_grp =df.groupby([idx]).y.mean().sort_values()
        self.feature_value_maps[idx] = {'map':dict(zip(df_grp.index,range(len(df_grp.index)))),
                                        'rev_map':dict(zip(range(len(df_grp.index)),df_grp.index))}
        return df[idx].map(self.feature_value_maps[idx]['map']).values
    
    def _format_feature(self,X,y,idx):
        if self.rev_feature_map[idx] in self.numeric_features:
            return X[:, idx], 'numeric'
        elif self.rev_feature_map[idx] in self.categorical_features:
            return self._format_categorical(X,y,idx) ,'categorical'
        else:
            raise ValueError("This tree supports only numeric or categorical features") 
        
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None,None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr ,best_split_type = None, None,None
        #print('initial gini: ',best_gini)
        for idx in self.feature_names.map(self.feature_map):
            xf,split_type = self._format_feature(X,y,idx)
            thresholds, classes = zip(*sorted(zip(xf, y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_split_type = split_type
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
            
        return best_idx, best_thr ,best_split_type

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        prob = num_samples_per_class[predicted_class]/np.sum(num_samples_per_class)
        node = Node(predicted_class=self.rev_taget_map[predicted_class],probability=prob)
        if depth < self.max_depth:
            #print('depth: ',depth)
            idx, thr,split_type = self._best_split(X, y)            
            #print(idx,split_type)
            if idx is not None:
                #indices_left = X[:, idx] < thr
                var ,_  = self._format_feature(X,y,idx) 
                #print(var)
                indices_left = var < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                if split_type == 'numeric':
                    node.threshold = thr
                    node.node_type = 'numeric'
                else :
                    #print(split_type)
                    #print(idx)
                    #print(thr)
                    #print(self.feature_value_maps[idx]['rev_map'].values())
                    thr_cat = pd.Series([x for x in self.feature_value_maps[idx]['rev_map'].keys() if x < thr])
                    node.threshold = thr_cat.map(self.feature_value_maps[idx]['rev_map']).values
                    node.node_type = 'categorical'
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs,ptype = 'class'):
        node = self.tree_
        while node.left:
            if node.node_type != 'categorical':
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                if inputs[node.feature_index] in node.threshold:
                    
                    node = node.left
                else:
                    node = node.right
        if ptype == 'class':
            return node.predicted_class
        elif ptype == 'prob' :
            if node.predicted_class == self.classes[0]:
                return [node.probability,1- node.probability]
            else:
                return [1- node.probability,node.probability,]
        else:
            return np.nan


