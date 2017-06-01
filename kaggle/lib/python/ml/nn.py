import numpy as np
import numpy.linalg
from numpy import newaxis, r_, c_, mat, e
import math
import matplotlib.pyplot as plt
import pandas as pd
from pooh.ml import linreg as lr

def cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd):
	m = X.shape[0]
	theta1 = np.array(nn_params[0:(input_layer_size+1) * hidden_layer_size])
	theta1 = theta1.reshape(hidden_layer_size, input_layer_size+1)
	theta2 = np.array(nn_params[(input_layer_size+1) * hidden_layer_size:])
	theta2 = theta2.reshape(num_labels, hidden_layer_size + 1)

	a1 = np.concatenate([np.ones(m).reshape(m, 1), X], axis = 1) 
	z2 = a1.dot(theta1.T)
	a2 = lr.sigmoid(z2)
	a2 = np.concatenate([np.ones(m).reshape(m,1), a2], axis = 1)
	z3 = a2.dot(theta2.T)
	htheta = lr.sigmoid(z3)
	#print htheta.shape

	np.savetxt('./dt/y.csv', y)
	J = 0.0
	for k in range(1, num_labels+1):
		yk = (y == k)
		#print yk.shape
		yk.dtype = np.dtype('i1')
		#print yk, ' ', yk.shape
		hthetak = htheta[:,k-1].reshape(m, 1)
		np.savetxt('./dt/hthetak-' + str(k) + '.csv', hthetak)
		#print hthetak
		# print yk.shape, ", ", np.log(hthetak).shape
		# asum = sum(-yk * np.log(hthetak) - (1 - yk) * np.log(1 - hthetak))
		#print 'yk[0] * np.log(hthetak): ', 'yk[0]: ', str(yk[0,0]), ', np.log(hthetak[0, 0]): ' + str(np.log(hthetak[0,0]))
		np.savetxt('./dt/yk-' + str(k) + '.csv', yk)
		np.savetxt('./dt/myk-' + str(k) + '.csv', -yk)
		hthetaklog =  np.log(hthetak) * -yk
		#print 'hthetak shape: ', hthetak.shape, ', yk shape: ', yk.shape, ', hthetaklog: ', hthetaklog.shape
		np.savetxt('./dt/hthetaklog-' + str(k) + '.csv', hthetaklog)
		#print hthetaklog
		hthetaklog2 =  np.log(1 - hthetak) * (1 - yk)
		np.savetxt('./dt/hthetaklog2-' + str(k) + '.csv', hthetaklog2)
		#print hthetaklog2
		asum = sum(-yk * np.log(hthetak) - (1 - yk) * np.log(1 - hthetak))
		print "asum: ", asum
		Jk = 1.0 / m * asum[0]
		J = J + Jk
		print('----------')

	grad = []
	regularization = lmbd / (2 * m) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2));
	#print 'lambda: %f, regularization: %f' % (lmbd, regularization)
	J = J + regularization;


	return J, grad

def sigmoid(X):
    '''Compute the sigmoid function '''
    #d = zeros(shape=(X.shape))

    den = 1.0 + math.e ** (-1.0 * X)
    d = 1.0 / den

    return d

def sigmoid_gradient(z):
    g = sigmoid(z) * (1 - sigmoid(z));
    return g

def rand_initialize_weights(l_in, l_out):
	epsilon_init = 0.12;
	W = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init;
	print W.shape
	return W




