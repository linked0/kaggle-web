import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pooh import linreg
from .. import utils
from .. import common

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m 
# Skip!

## ======================= Part 2: Plotting =======================
print('Plotting Data ...')
folder = utils.get_folder(__file__)
data = np.loadtxt(folder + '/' + 'ex1data1.txt', delimiter=',')

fig = plt.figure()
plt.plot(data[:,0], data[:,1], '.')

input('Program paused. Press enter to continue.'); # raw_input in Python 2.7

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
m = data.shape[0]
newdt0 = np.ones(m).reshape(m,1)
X = np.concatenate((newdt0, data[:,0].reshape(m,1)), axis=1)
y = data[1]
theta = np.zeros(X.shape[1])
common.compute_cost(X, y, theta)

