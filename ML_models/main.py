# -*- coding: utf-8 -*-
"""
data: train data is being imported from Project.py file
data_classification: function to run various classifiers and applying feature-extraction techniques
Y_train: class labels of train data set
Y_test: Class labels of cross validation set
lr: Logistic Regression Classifier
nb: Naive Bayes Classifier
dt = Decision Tree Classifier
rf: Random Forest Classifier
knn: k-Nearest Neighbor Classifier
tf-idf: Tf-idf Feature extraction technique
bow: Bag of Words Feature extraction technique
ngram12: Bag of Words with N-gram of range (1,2)
ngram23: Bag of Words with N-gram of range (2,3)
ngram13: Bag of Words with N-gram of range (1,3)

Output of project.py given the the confusion matrix and classification report
of the desired classifier.
The final output is the creation of testing_data_labelfile.txt file having
predicted class labels of text data.
"""

#from main import data

from project import data,data_classification,Y_train,Y_test

clf = data_classification(clf_opt='lr')
clf.classification('ngram13', Y_train, Y_test)