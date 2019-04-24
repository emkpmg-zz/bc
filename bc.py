# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:59:33 2019

@author: PIANDT
"""

#import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 01,02 Load dataset, explore and identify relevant variables / labelling
data = load_breast_cancer()

#dictionary keys to consider: 
#Classification label names (target_names) --> Actual labels (target)
#Attribute/feature names (feature_names) --> Attributes (data)

outputLabels = data['target_names']
output = data['target']
inputLabels = data['feature_names']
input = data['data']

print(outputLabels) #this is a binary classification do its either 0 or 1

#  malignant-1, benign-0
print(output[20])
print(inputLabels[0])
print(input[0])

# 03 Data Split --> Training, Validation and Test set
#random_state - int or RandomState, Pseudo-random number generator state used for random sampling
trainData, testData, trainDataLabels, testDataLabels = train_test_split(input,output,test_size=0.35,random_state=42)

# 04 module selection for binary classification tasks --> Naive Bayes (NB)
# import GaussianNB module. # initialize model with the GaussianNB() function 
# train model by fitting to data --> gnb.fit()

# Initialize our classifier
binaryClassifier = GaussianNB()

# Train our classifier
modelTrain = binaryClassifier.fit(trainData, trainDataLabels)

# 05 Predictions
prediction = binaryClassifier.predict(testData)
print(prediction) #prints out 0s and 1s to represent benign or malignant from test dataset

# 06 Model accuracy - comparing the two arrays (testDataLabels vs. prediction)
# Use accuracy_score() fxn to determine accuracy of our ml classifier.

print(accuracy_score(testDataLabels, prediction))

