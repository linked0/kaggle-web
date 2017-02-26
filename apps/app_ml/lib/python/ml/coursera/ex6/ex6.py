## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines

import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from pooh import mllib as ml
import matplotlib.pyplot as plt
import ex6func

## Initialization
import os
clear = lambda: os.system('clear')
clear()

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment
data = sio.loadmat('ex6data1.mat');
X1 = data['X'][:,0]
y = data['y']
y = np.concatenate(y, axis = 0)
X1_0 = X1[y==0]
X1_1 = X1[y==1]
X2 = data['X'][:,1]
X2_0 = X2[y==0]
X2_1 = X2[y==1]
m = X.shape[0]

# Plot training data
fig = plt.figure()
plt.plot(X1_0, X2_0, 'yo', markersize=7);
plt.plot(X1_1, X2_1, 'r+', markersize=7);

raw_input('Program paused. Press enter to continue.');

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#
printf('\nTraining Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#


## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#


## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#



## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 


