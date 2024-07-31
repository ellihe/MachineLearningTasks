# 1NN classifier for course excercises
# author: Ellinoora Hetemaa

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from sklearn.metrics import accuracy_score


# Classifies inputvector x with nearest neigbour method. Checks which image in
# training set trdata is closest to x (in manhattan terms) and returns its
# class label
def cifar10_classifier_1nn(x, trdata, trlabels):
    dists = []
    for trd in trdata:
        # d = np.sum((x - trd)**2)     # Euclidean distance, just without sqrt
        d = np.sum(np.absolute(x - trd))  # Manhattan distance
        dists.append(d)
    dists = np.array(dists)
    mindist_ind = np.where(dists == np.min(dists))
    return trlabels[mindist_ind]


def class_acc(pred, gt):
    return 100*accuracy_score(gt, pred), "%"


def cifar10_classifier_random(x):
    return np.random.randint(0, 10)


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


#datadict = unpickle('cifar-10-batches-py/data_batch_1')
#datadict = unpickle('test_batch')

#X = datadict["data"]
#Y = datadict["labels"]

#print(X.shape)

#labeldict = unpickle('cifar-10-batches-py/batches.meta')
#label_names = labeldict["label_names"]

#X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
#Y = np.array(Y)

#p = class_acc(Y, Y)
#print(p[0], p[1])

#randlabels = []
#for i in range(len(X)):
#    randlabels.append(cifar10_classifier_random(X[i]))
#acc = class_acc(randlabels, Y)
#print("The accurracy with random classifier for the test batch is", acc[0], acc[1])

# Testing 1NN classifier for all the test data and all 5 training batches

# Loads the train data
for i in range(1, 6):
    if i == 1:
        datadict = unpickle('cifar-10-batches-py/data_batch_{}'.format(i))
        X = datadict["data"]
        Y = datadict["labels"]
    else:
        datadict2 = unpickle('cifar-10-batches-py/data_batch_{}'.format(i))
        new_Y = np.concatenate((Y, datadict2["labels"]), 0)
        new_X = np.concatenate((X, datadict2["data"]), 0)
        X = new_X
        Y = new_Y

X = X.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")

# Loads test data
datadict_test = unpickle('cifar-10-batches-py/test_batch')
X_test = datadict_test["data"]
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("intD32")
Y_test = datadict_test["labels"]

# Calculating accurracy for the first n images, n in the range(1, 10 000)
n = 100
lablist = []
for x in X_test[0:n]:
    lab = cifar10_classifier_1nn(x, X, Y)
    lablist.append(lab)

acc = class_acc(lablist, Y[0:n])
print("The accuracy for the test batch with 1NN classifier is", acc)

