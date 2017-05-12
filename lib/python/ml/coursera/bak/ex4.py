import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from pooh.ml import nn as nn 

input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden Images of Digits
num_labels = 10         # 10 labels, from 1 to 10

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

data = sio.loadmat('ex4data1.mat')

array_a = np.random.rand(10,2)
array_b = np.random.permutation(range(10))

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

data2 = sio.loadmat('ex4weights.mat')

nn_params = list(np.array(data2['Theta1']).reshape(-1,))
nn_params += list(np.array(data2['Theta2']).reshape(-1,))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print ('\nFeedforward Using Neural Network2 ...\n')

# Weight regularization parameter (we set this to 0 here)
lmbd = 0

X = np.array(data['X'])
y = np.array(data['y'])
#y[y==10] = 0

J, grad = nn.cost_func(nn_params, input_layer_size, hidden_layer_size,  num_labels, X, y, lmbd);

print 'Cost at parameters (loaded from ex4weights): %f (this value should be about 0.287629\n' % (J);
print 'grad: ', grad
raw_input('Program paused. Press enter to continue.')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')
# Weight regularization parameter (we set this to 1 here).
lmbd = 1.0;

J, grad = nn.cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)

print 'Cost at parameters (loaded from ex4weights): %f (this value should be about 0.383770)\n' % (J)
print 'Grad: ', grad

raw_input('Program paused. Press enter to continue.\n');

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = nn.sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]));
print 'Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]: ', g
raw_input('Program paused. Press enter to continue.\n');

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = nn.rand_initialize_weights(input_layer_size, hidden_layer_size);
initial_Theta2 = nn.rand_initialize_weights(hidden_layer_size, num_labels);

# Unroll parameters
initial_nn_params = list(np.array(initial_Theta1).reshape(-1,))
initial_nn_params += list(np.array(initial_Theta2).reshape(-1,))
# print initial_nn_params
