import numpy as np
import numpy.linalg 
from numpy import newaxis, r_, c_, mat, e
import math
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats

def linreg_cost_func(X, y, theta, lmbd):
    J = 0
    m = X.shape[0]
    n = X.shape[1]
    theta = theta.reshape(n,1)
    grad = np.zeros(theta.shape)
    htheta = X.dot(theta)
   
    J = 1.0/(2*m) * np.sum((htheta-y)**2) + float(lmbd)/(2*m) * np.sum(theta[1:]**2)
 
    return J

def linreg_grad_func(X, y, theta, lmbd):
    m = X.shape[0]
    n = X.shape[1]
    theta = theta.reshape(n,1)
    grad = np.zeros(theta.shape)
    htheta = X.dot(theta)
   
    grad = 1.0/m * X.T.dot(htheta-y)
    grad[1:] = grad[1:] + float(lmbd) / m * theta[1:]
 
    return grad

def linreg_cost_func_opt(theta, X, y):
    J = 0
    m = X.shape[0]
    n = X.shape[1]
    print "X shape: ", X.shape
    print "Y shape: ", y.shape
    print "theta shape: ", theta.shape 
    theta = theta.reshape(n,1)
    grad = np.zeros(theta.shape)
    htheta = X.dot(theta)
   
    J = 1.0/(2*m) * np.sum((htheta-y)**2)
    print "J shape: ", J.shape
 
    return J

def linreg_grad_func_opt(theta, X, y):
    print "linreg_grad_func_opt"
    m = X.shape[0]
    n = X.shape[1]
    theta = theta.reshape(n,1)
    grad = np.zeros(theta.shape)
    htheta = X.dot(theta)
   
    grad = 1.0/m * X.T.dot(htheta-y)
 
    grad = grad.reshape(-1)
    print "linreg_grad_func_opt end: ", grad.shape
    return grad

def train_linreg(X, y, lmbd):
    initTheta = np.ones(2)
    theta = opt.fmin_ncg(linreg_cost_func_opt, initTheta, 
        fprime=linreg_grad_func_opt, args=(X, y))

    print 'train_linreg'
    return theta


def pearson2(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1
    
    if len(si) == 0:
        return 0

    prefs_p1 = [prefs[p1][item] for item in si.keys()]
    prefs_p2 = [prefs[p2][item] for item in si.keys()]

    return stats.pearsonr(prefs_p1, prefs_p2)[0]

def pearson1(p1, p2):
    return stats.pearsonr(p1, p2)[0]

def pearson(v1, v2):
  # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)

  # Sums of the squares
    sum1Sq = sum([pow(v, 2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])

  # Sum of the products
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

  # Calculate r (Pearson score)
    num = pSum - sum1 * sum2 / len(v1)
    den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2)
               / len(v1)))
    if den == 0:
        return 0

    return 1.0 - num / den


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)