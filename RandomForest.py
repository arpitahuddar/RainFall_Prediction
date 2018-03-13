#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:42:34 2017

@author: ahuddar
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
 
#import pdb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
# File Paths
#INPUT_PATH = "../inputs/breast-cancer-wisconsin.data"
#OUTPUT_PATH = "../inputs/breast-cancer-wisconsin.csv"
 
# Headers
#HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
 #          "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]

HEADERS = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5","Feature6", "Feature7", "Feature8", "Feature9", "Feature10","Feature11", "Feature12", "Feature13"]
 
def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return:
    """

    data = pd.read_txt('data/X_train.txt')
    target = pd.read_txt('data/Y_train.txt')
    return data
 
 
def get_headers(dataset):
    """
    dataset headers
    :param dataset:
    :return:
    """
    return dataset.columns.values
 
 
def add_headers(dataset, headers):
    """
    Add the headers to the dataset
    :param dataset:
    :param headers:
    :return:
    """
    dataset.columns = headers
    return dataset
 
 
def data_file_to_csv():
    """
 
    :return:
    """
 
    # Headers
    headers = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5","Feature6", "Feature7", "Feature8", "Feature9", "Feature10","Feature11", "Feature12", "Feature13", "Feature14",]
    # Load the dataset into Pandas data frame
    dataset = read_data('data/X_train.txt')
    # Add the headers to the loaded dataset
    dataset = add_headers(dataset, headers)
    # Save the loaded dataset into csv format
    dataset.to_csv('RandonForest_csv', index=False)
    print "File saved ...!"
 
 
def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """
 
    # Split dataset into train and test dataset
    train_x, test_x,train_y, test_y = train_test_split(dataset[feature_headers],train_size=train_percentage)
    return train_x, test_x, train_y, test_y

 
def handel_missing_values(dataset, missing_values_header, missing_label):
    """
    Filter missing values from the dataset
    :param dataset:
    :param missing_values_header:
    :param missing_label:
    :return:
    """
 
    return dataset[dataset[missing_values_header] != missing_label]
 
 
def random_forest_regressor(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestRegressor()
    clf.fit(features, target)
    return clf
 
 
def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print dataset.describe()
 
 
def main():
    """
    Main function
    :return:
    """
    #X = np.genfromtxt('data/X_train.txt', delimiter=None)
    #Y = np.genfromtxt('data/Y_train.txt', delimiter=None)
    # Load the csv file into pandas dataframe
    
    dataset = pd.read_csv('data/X_train.txt', sep=" ", header=None)
    target = pd.read_csv('data/Y_train.txt', sep=" ", header=None)
    testdata = pd.read_csv('data/X_test.txt', sep=" ", header=None)

    #data.columns = ["a", "b", "c", "etc."]
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)
 
    # Filter missing values
    #dataset = handel_missing_values(dataset, HEADERS[11], 0)
    #dataset_statistics(dataset)
    train_x, test_x,train_y, test_y = train_test_split(dataset,target,test_size=0.33, random_state=42)
    
    #train_y, test_y = train_test_split(target, 0.7)
 
    # Train and Test dataset size details
    print "Train_x Shape :: ", train_x.shape
    print "Train_y Shape :: ", train_y.shape
    print "Test_x Shape :: ", test_x.shape
    print "Test_y Shape :: ", test_y.shape
 
    # Create random forest classifier instance
    #trained_model = random_forest_regressor(train_x, train_y)
    #print "Trained model :: ", trained_model
    #predictions = trained_model.predict(test_x)
 
    #print predictions
    
    #for i in range(0,100):
    #    print predictions[i]
  #  for i in range(0,500):
  #       print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])

    #predictions = trained_model.predict(testdata)
    
    #print predictions
    #Yte = np.vstack((np.arange(testdata.shape[0]), predictions)).T
    #np.savetxt('data/Y_submit_Random_forest.txt', Yte, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')
    
    '''print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
#   print "Test Accuracy  :: ", accuracy_score(test_y, predictions)
    print " Confusion matrix ", confusion_matrix(test_y, predictions)
 
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('ROC Curve')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()'''
    
 
if __name__ == "__main__":
    main()