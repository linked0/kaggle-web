import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pooh.ml import linreg

data = pd.read_csv('ex1data1.txt', header=None)

fig = plt.figure()
plt.plot(data[0], data[1], '.')

X = pd.concat([pd.Series(np.ones(len(data))), data[0]], axis=1)
y = data[1]
theta = np.zeros(X.shape[1])
linreg.compute_cost(X, y, theta)

theta2 = linreg.gradient_descent(X, y, theta, 0.01, 1500)
plt.plot(X[1], X.dot(theta2))
plt.legend(['Training data', 'Linear regression'], 'right')

predict1 = [1, 3.5]
print "For population = 35000, we predict a profit of %f" % (np.dot(theta2, np.transpose(predict1))*10000)
predict2 = [1, 7.0]
print "For population = 70000, we predict a profit of %f" % (np.dot(theta2, np.transpose(predict2))*10000)

