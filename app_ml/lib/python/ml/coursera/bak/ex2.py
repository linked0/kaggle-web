import numpy as np
from pooh.ml import linreg
import pandas as pd
import numpy.linalg
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('./ex2data1.txt', header=None)
X = data[:][[0,1]]
pos = (data[2] == 1)
neg = (data[2] == 0)
fig = plt.figure()
plt.plot(X[pos][0], X[pos][1], 'b+')
plt.plot(X[neg][0], X[neg][1], 'yo')

plt.legend(['Admitted', 'Not admitted'], loc='right')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')

X = pd.concat([Series(np.ones(X.shape[0])), X], axis=1)
X.columns = [0,1,2]
y = data[2]

theta = np.zeros(X.shape[1])

theta2 = opt.fmin_bfgs(linreg.cost_func, theta, fprime=linreg.grad_func, args=(X, y))
print 'new theta: ' + str(theta2)

linreg.plot_decision_boundary(theta2, X, y)
