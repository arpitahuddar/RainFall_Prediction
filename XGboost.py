#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:47:27 2017

@author: ahuddar
"""


import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

# Data Loading
X = np.genfromtxt('data2/X_train.txt', delimiter=None)
Y = np.genfromtxt('data2/Y_train.txt', delimiter=None)

Xte = np.genfromtxt('data2/X_test.txt', delimiter=None)

Xsample, Ysample = X[:120000], Y[:120000]

X_all, Y_all = X[:200000], Y[:200000]

#XtS, params = ml.rescale(Xt) # Normalize the features
#XvS, _ = ml.rescale(Xv, params) # Normalize the features

X_train, X_test, y_train, y_test = train_test_split(Xsample, Ysample, test_size=0.2, random_state=42)

# use DMatrix for xgbosot
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
drealTest = xgb.DMatrix(Xte)

dtrainall = xgb.DMatrix(X_all, label=y_train)
dtestall = xgb.DMatrix(X_all, label=y_test)

# use svmlight file for xgboost
dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

print dump_svmlight_file

# set xgboost params
param = {
    'max_depth': 25,  # the maximum depth of each tree
    'eta': 0.01,  # the training step for each iteration
    'silent': 0,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 2 # the number of classes that exist in this datset
    
    }  
#'tree_method':'exact'
num_round = 30  # the number of training iterations

#------------- numpy array ------------------
# training and testing - numpy matrices

bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

best_preds_training = np.array([np.argmax(line) for line in bst.predict(dtrain)])
best_preds_training_all = np.array([np.argmax(line) for line in bst.predict(dtrainall)])

# extracting most confident predictions
#best_preds = np.asarray([np.argmax(line) for line in preds])
#print "Numpy array precision:", precision_score(y_test, best_preds, average='macro')

#print preds
#print best_preds

#print "roc_auc_score",roc_auc_score(y_test,best_preds,average='macro')
# ------------- svm file ---------------------
# training and testing - svm file
#bst_svm = xgb.train(param, dtrain_svm, num_round)
#preds = bst_svm.predict(dtest_svm)

# extracting most confident predictions
#best_preds_svm = [np.argmax(line) for line in preds]
#print "Svm file precision:",precision_score(y_test, best_preds_svm, average='macro')
#print "roc_auc_score for svm",roc_auc_score(y_test,best_preds_svm,average='macro')
#print best_preds_svm
# --------------------------------------------

#preds = bst_svm.predict(drealTest)


#print preds
#Yte = np.vstack((np.arange(Xte.shape[0]), preds[:,1])).T
#np.savetxt('data/Y_submit_ful_data_numpy_xgb.txt', Yte, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')

# dump the models
#bst.dump_model('dump.raw.txt')
#bst_svm.dump_model('dump_svm.raw.txt')


# save the models for later
#joblib.dump(bst, 'bst_model.pkl', compress=True)
#joblib.dump(bst_svm, 'bst_svm_model.pkl', compress=True)
#print Xte.shape[0]
#print preds[:,1]


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
#fpr, tpr, _ = roc_curve(y_test, best_preds_svm)
fpr, tpr, _ = roc_curve(Y_all, best_preds_training_all)
#fpr, tpr, _ = roc_curve(y_train, bst_svm.predict(dtrain))
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test, best_preds_svm)
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


for i in maxDepth:
bst = xgb.train(param, dtrain, num_round)
auc_T.append(learner.auc(Xt,Yt))
auc_V.append(learner.auc(Xv,Yv))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='(AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


