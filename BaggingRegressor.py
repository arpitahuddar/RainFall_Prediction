#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:01:19 2017

@author: ahuddar
"""
# Bagged Decision Trees for Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np

X = np.genfromtxt('data2/X_train.txt', delimiter=None)
y = np.genfromtxt('data2/Y_train.txt', delimiter=None)

Xte = np.genfromtxt('data2/X_test.txt', delimiter=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeRegressor()
num_trees = 100
model = BaggingRegressor(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train,y_train , cv=kfold)
print(results.mean())


from sklearn.metrics import roc_auc_score

print "roc_auc_score for training data",roc_auc_score(y_train,model.predict(X_train),average='macro')
print "roc_auc_score for testing data",roc_auc_score(y_test,model.predict(X_test),average='macro')