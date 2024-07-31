import numpy as np

print("Reading data...")
X_train = np.loadtxt("X_train.dat")
Y_train = np.loadtxt("y_train.dat")

X_test = np.loadtxt("X_test.dat")
Y_test = np.loadtxt("y_test.dat")
print("Done!")


def my_linfit(x, y):
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    a = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x*x) - np.power(np.sum(x), 2))
    b = (sum(y)-sum(a*x))/n
    return a, b

print("Shape of training data (404, 13)")
print("Computing 1-dim linear regression")
# Let's train the model with only one of the possible 13 inputs at time
for i in range(0, 13):
    # a and b of the trained regression model
    a, b = my_linfit(X_train[:, i], Y_train)

    # Let's test our model with test data
    X_vector = X_test[:, i]
    y_pred = a*X_vector + b
    y_actual = Y_test
    MAE = 0
    N = len(Y_test)
    for k in range(0, N - 1):
        MAE += np.abs(y_pred[k] - y_actual[k])
    MAE = MAE/N
    print("Lin. reg. dim", i+1, "accuracy:", MAE)

