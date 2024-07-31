# Bayesian classification with accuracy calculation
# author: Ellinoora Hetemaa

import pickle

import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import scipy as sc
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal


def class_acc(pred, gt):
    # Compute accuracy for each possible representation
    sum_of_same_labels = 0
    i = 0
    for label in pred:
        if gt[i] == label:
            sum_of_same_labels += 1
        i += 1

    return 100*sum_of_same_labels/len(pred), "%"


def cifar10_2x2_color(x):
    rows = len(x)
    subimg_n = int(3 * (2 ** 2))
    f = []
    for item in x:
        for j in range(0, subimg_n):
            pieces = np.mean(item[j * 3072 // subimg_n:(j + 1) * 3072 // subimg_n], axis=0, dtype=np.float64)
            f.append(pieces)
    f = np.array(f, dtype=np.float64)
    f = np.reshape(f, (rows, subimg_n))
    return f

def cifar10_NxN_color(x, N):
    rows = len(x)
    subimg_n = int(3 * (N ** 2))
    f = []
    for item in x:
        for j in range(0, subimg_n):
            pieces = np.mean(item[j * 3072 // subimg_n:(j + 1) * 3072 // subimg_n], axis=0, dtype=np.float64)
            f.append(pieces)
    f = np.array(f, dtype=np.float64)
    f = np.reshape(f, (rows, subimg_n))
    return f

def cifar_10_bayes_learn(Xf, Y):
    mus = []
    covs = []
    p = []
    for j in range(0, 10):
        indexes_of_class_j = []
        i = 0
        for label in Y:
            if label == j:
                indexes_of_class_j.append(i)
            i += 1
            #print(j)

        i = 0

        arrayOfColors = []
        for color in Xf:
            if i in indexes_of_class_j:
                arrayOfColors.append(color)
            i += 1

        # Mean colors and variances of each class in cifar10 dataset
        mus.append(np.mean(arrayOfColors, axis=0, dtype=np.float64))
        covs.append(np.cov(arrayOfColors, rowvar=0))
        prior = len(indexes_of_class_j)/len(Xf)
        p.append(prior)
    return mus, covs, p


def cifar10_classifier_bayes(x, mu, cov, p, size):

    max_numerator = 0
    class_of_x = -1
    for i in range(0, 10):
        normal = multivariate_normal.pdf(x, mu[i], cov[i])
        numerator = normal * p[i]
        if numerator > max_numerator:
            max_numerator = numerator
            class_of_x = i

    return class_of_x


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


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

datadict_test = unpickle('cifar-10-batches-py/test_batch')
X_test = datadict_test["data"]
Y_test = datadict_test["labels"]

nums = [1, 2, 4]
accuracies = []
for num in nums:
    pred = []
    k = 0
    Xp = cifar10_NxN_color(X, num)
    Xp_test = cifar10_NxN_color(X_test, num)
    mu, sigma, p = cifar_10_bayes_learn(Xp, Y)
    for x in Xp_test:
        class_of_x = cifar10_classifier_bayes(x, mu, sigma, p, num)
        pred.append(class_of_x)
        k += 1
    accuracy = class_acc(pred, Y_test)
    accuracies.append(accuracy[0])
    print("accuracy for size", num, "is: ", accuracy)

plt.plot(nums, accuracies)
plt.ylim(0, 100)
plt.xlim(0, 16)
plt.show()
