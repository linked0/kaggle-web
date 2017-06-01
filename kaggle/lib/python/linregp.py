import numpy as np
import numpy.linalg 
from numpy import newaxis, r_, c_, mat, e
import math
import matplotlib.pyplot as plt
import pandas as pd

####################################################
# the order of the parameters of X, y, theta 
####################################################

def compute_cost(X, y, theta):
    J = 0
    h = 0

    m = len(y)

    for i in range(m):
        h = h + (X.ix[i].values.dot(theta.T) - y[i])**2

    J = 0.5*h/m
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = []
    m = len(y)
    theta2 = theta.copy()

    for iter in range(num_iters):
        g = np.zeros(m)
        p = np.zeros(len(theta2))

        for i in range(m):
            sum = X.ix[i].values.dot(theta2.T)
            g[i] = sum

        for j in range(len(theta2)):
            for i in range(m):
                p[j] = p[j] + (g[i] - y[i]) * X.ix[i, j]
            theta2[j] = theta2[j] - alpha*p[j]/m

#        print compute_cost(X, y , theta2)

    return theta2

def feature_normalize(X):
    mu = X.mean()
    sigma = X.std()
    normal_X = (X - mu) / sigma
    return normal_X, mu, sigma
    
def normal_eqn(X, y):
    return linalg.inv(X.T.dot(X)).dot(X.T).dot(y)    
			
def sigmoid(X):
    '''Compute the sigmoid function '''
    #d = zeros(shape=(X.shape))

    den = 1.0 + math.e ** (-1.0 * X)
    d = 1.0 / den

    return d

def cost_func(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    cost = -y*np.log(h) - (1-y)*np.log(1-h)
    return cost.mean()

def grad_func(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    error = h - y
    grad = np.dot(error, X)/len(y)
    return grad 

def plot_data(X, y):
    fig = plt.figure()
    pos = (y == 1)
    neg = (y == 0)
    plt.plot(X[pos][0], X[pos][1], 'b+')
    plt.plot(X[neg][0], X[neg][1], 'yo')
    plt.legend(['Admitted', 'Not admitted'], loc='right')

def map_feature(X1, X2):
    degree = 6
    X = pd.Series(np.ones(X1.shape[0]))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            X = pd.concat([X, X1**(i-j) * X2**(j)], axis=1, ignore_index=True)
    return X

def plot_decision_boundary(theta, X, y):
    plot_data(X, y, theta)

def opt_bfgs(cost_f, theta, grad_f, X, y):
    return opt.fmin_bfgs(linreg.cost_func, theta, fprime=linreg.grad_func, args=(X, y))

def cost_func_reg(theta, X, y, lmd):
    m = X.shape[0]
    # print "theta: ", theta
    # print "y: ", y, "shape of y: ", y.shape ,  "type of y: " , type(y)

    predictions = sigmoid(X.dot(c_[theta]))
    predictions = predictions.T.values[0]
    # print "predictions: ", predictions
    # print "type of predictions: ", type(predictions)

    # print "np.log(predictions): ", np.log(predictions)
    # print "-y*np.log(predictions): ", -y*np.log(predictions)
    # print 'np.log(1-predictions): ', np.log(1-predictions)

    J = (-y*np.log(predictions)) - (1-y)*(np.log(1-predictions))
    # print "mean of J: ", J
    J = J.mean()
    J_reg = lmd/(2*m) * (theta[1:]**2).sum()

    J += J_reg
    return J 

def opt_powell(init_theta, X, y, lmd, cost_func):
    option = {'full_output': True}
    theta, cost, _, _, _, _ = opt.fmin_powell(lambda t: cost_func(t, X, y, lmd), init_theta, **options)
    
    return theta         

