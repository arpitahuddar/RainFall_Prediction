#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:32:51 2017

@author: ahuddar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score

# Create a dataset

X = np.genfromtxt('data/X_train.txt', delimiter=None)
y = np.genfromtxt('data/Y_train.txt', delimiter=None)

Xte = np.genfromtxt('data/X_test.txt', delimiter=None)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=4)

"""max_depth = 30

#regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
#y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

y_rf_test = regr_rf.predict(Xte)

print "roc_auc_score for training data",roc_auc_score(y_train,regr_rf.predict(X_train),average='macro')
print "roc_auc_score for testing data",roc_auc_score(y_test,y_rf,average='macro')

print y_rf_test

Yte = np.vstack((np.arange(Xte.shape[0]), y_rf_test)).T
np.savetxt('data/Y_submit_ful_randomForest_my_data.txt', Yte, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')
"""

regr_rf = RandomForestRegressor(max_depth=20, random_state=2)
regr_rf.fit(X_train, y_train)
y_rf_test = regr_rf.predict(Xte)
print roc_auc_score(y_test,y_rf_test,average='macro')
'''    
result=[]
i=0
depth_values=[10,15,20,25,30,40,45]
min_leaf=[2,5,7,9,10,11,13]
min_split=[7,9,10,12,14,16,18]
for i in range(0,7,1):
    regr_rf = RandomForestRegressor(max_depth=depth_values[i], random_state=2)
    regr_rf.fit(X_train, y_train)
    y_rf_test = regr_rf.predict(Xte)
    result.append(roc_auc_score(y_test,y_rf_test,average='macro'))

plt.figure(2)
plt.plot(depth_values,result)
plt.show

i=0
result=[]
for i in range(0,7,1):
    regr_rf = RandomForestRegressor(min_samples_split=min_split[i], random_state=2)
    regr_rf.fit(X_train, y_train)
    y_rf_test = regr_rf.predict(Xte)
    result.append(roc_auc_score(y_test,y_rf_test,average='macro'))

plt.figure(3)
plt.plot(min_split,result)
plt.show()

i=0
result=[]
for i in range(0,7,1):
    regr_rf = RandomForestRegressor(min_samples_leaf=min_leaf[i], random_state=2)
    regr_rf.fit(X_train, y_train)
    y_rf_test = regr_rf.predict(Xte)
    result.append(roc_auc_score(y_test,y_rf_test,average='macro'))

plt.figure(4)
plt.plot(depth_values,result)
plt.show()

'''


'''
# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
            c="navy", s=s, marker="s", alpha=a, label="Data")
#plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
#            c="cornflowerblue", s=s, alpha=a,
 #          label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
            c="c", s=s, marker="^", alpha=a,
            label="RF score=%.2f" % regr_rf.score(X_test, y_test))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()'''