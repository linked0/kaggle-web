import pandas as pd
import numpy as np
from pooh.ml import linreg
from pandas import Series

data = pd.read_csv('ex1data2.txt', header=None)
X = data.ix[:, 0:1]
y = data[2]
X, mu, sigma = linreg.feature_normalize(X)
X = pd.concat([Series(np.ones(X.shape[0])), X], axis = 1)
X.columns = [0,1,2]
theta = np.zeros(X.shape[1])
theta = linreg.gradient_descent(X, y, theta, 0.01, 400)

print "Result of gradient_descent: " + str(theta)

print "price of 1650 sq-ft, 3 bd"
t = [1650, 3]
t = [1] + list((t-mu.values)/sigma.values)
print t
print type(t)
print theta
price = np.dot(t, list(theta))
print price

print "Start normal equation"
data = pd.read_csv('ex1data2.txt')
