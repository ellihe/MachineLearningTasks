import numpy

print("Reading data...")
X_train = numpy.loadtxt("X_train.dat")
Y_train = numpy.loadtxt("y_train.dat")

X_test = numpy.loadtxt("X_test.dat")
Y_test = numpy.loadtxt("y_test.dat")
print("Done!")
print("Computing baseline regression.")
mean_absolute_error = 0
N = len(Y_train)
for k in range(0, N-1):
    mean_absolute_error += numpy.abs((Y_train[k], numpy.mean(X_train[k])))

mean_absolute_error = 1/N
print(mean_absolute_error)
