# Calculates roc curve for certain dataset
# author: Ellinoora Hetemaa

import pickle

import numpy as np
import matplotlib
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

testLabels = 1 - np.asarray(np.genfromtxt('detector_groundtruth.dat'))
output = np.asarray(np.genfromtxt('detector_output.dat'))

print(testLabels)
print(output)
# the confidence score of positive class(assuming class 1 be positive class, and 0 be negative)
fpr, tpr, _ = roc_curve(testLabels, output)
# create ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
