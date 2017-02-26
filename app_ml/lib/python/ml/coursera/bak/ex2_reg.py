## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#  
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#  
#  You will need to complete the following functions in this exericse:
#  
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#     
#  For this exercise, you will not need to change any code in this file,
# or any other files other than those mentioned above.
#

import pandas as pd
import numpy as np
from pooh.ml import linreg as lr

data = pd.read_csv('ex2data2.txt', header=None)
X = data[[0, 1]]
y = data[2]   
lr.plot_data(X, y)

X = lr.map_feature(X[0], X[1])

lmd = 1
print "X.shape[1]: ", X.shape[1]
theta = np.zeros(X.shape[1])
cost = lr.cost_func_reg(theta, X, y, lmd)
print "cost: %f" % cost

