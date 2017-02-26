import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from pooh import mllib as ml
import matplotlib.pyplot as plt

# SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
# algorithm. 
#    [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
#    SVM classifier and returns trained model. X is the matrix of training 
#    examples.  Each row is a training example, and the jth column holds the 
#    jth feature.  Y is a column matrix containing 1 for positive examples 
#    and 0 for negative examples.  C is the standard SVM regularization 
#    parameter.  tol is a tolerance value used for determining equality of 
#    floating point numbers. max_passes controls the number of iterations
#    over the dataset (without changes to alpha) before the algorithm quits.
# 
#  Note: This is a simplified version of the SMO algorithm for training
#        SVMs. In practice, if you want to train an SVM classifier, we
#        recommend using an optimized package such as:  
# 
#            LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
#            SVMLight (http://svmlight.joachims.org/)
# 
# 
def svmTrain(X, y, C, kernelFunction, tol, max_passes):

# LINEARKERNEL returns a linear kernel between x1 and x2
#    sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
#    and returns the value in sim
def linearKernel(x1, x2):
	# Ensure that x1 and x2 are column vectors
	#x1 = x1(:); x2 = x2(:);

	# Compute the kernel
	#sim = x1' * x2;  # dot product

	return 0

