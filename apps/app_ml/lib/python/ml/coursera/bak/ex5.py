## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance

import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from pooh import mllib as ml
import matplotlib.pyplot as plt

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = sio.loadmat ('ex5data1.mat');

# m = Number of examples
X = data['X']
y = data['y']
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'r+')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

# raw_input('Program paused. Press enter to continue.\n');

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#
theta = np.ones(2)
X0 = np.ones(m).reshape(m,1)
X = np.concatenate([X0, X], axis=1)
J = ml.linreg_cost_func(X, y, theta, 1)

print 'Cost at theta = [1 ; 1]: %f' % (J)
print '(this value should be about 303.993192)\n'

raw_input('Program paused. Press enter to continue.\n');

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

grad = ml.linreg_grad_func(X, y, theta, 1)

print 'Gradient at theta = [1 ; 1]:  [%f; %f] \n(this value should be about [-15.303016; 598.250744])' % (grad[0], grad[1]);

# raw_input('Program paused. Press enter to continue.\n');

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
lmbd = 0;
theta = ml.train_linreg(X, y, lmbd);
print "train result: ", theta

#  Plot fit over the data
# plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
# xlabel('Change in water level (x)');
# ylabel('Water flowing out of the dam (y)');
# hold on;
# plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
# hold off;

# fprintf('Program paused. Press enter to continue.\n');
# pause;
