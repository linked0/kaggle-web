#!/usr/bin/env python

__author__ = 'linked0'

import logging
import sys

import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import *

import common.strings as strs
import ml_algs.model_data as mdata
from gui.ml_analysis import MainWindow as mlm
from gui.house_prices import MainWindow as housem
from ml_algs.decision_tree import AlgDecisionTree
from ml_algs.ensemble import AlgEnsemble
from ml_algs.gaussian_nb import AlgGaussianNB
from ml_algs.kernel_svm import AlgKernelSVM
from ml_algs.kneighbors import AlgKNeighbors
from ml_algs.lin_reg import AlgLinearRegression
from ml_algs.log_reg import AlgLogisticRegression
from ml_algs.neural_net import AlgNeuralNet
from ml_algs.random_forest import AlgRandomForest
from ml_algs.svm import AlgSVM



logging.basicConfig(format=strs.log_format,level=logging.DEBUG,stream=sys.stderr)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    print('main start:{0}, e:{1}'.format("hello", "hi"))
    mdata.init()
    print('before mdata.alg_kernel_svm')
    mdata.set_algorithm_obj(mdata.alg_kernel_svm, AlgKernelSVM())
    print('before mdata.alg_k_neighbor')
    mdata.set_algorithm_obj(mdata.alg_k_neighbor, AlgKNeighbors())
    print('before mdata.alg_linear_reg')
    mdata.set_algorithm_obj(mdata.alg_linear_reg, AlgLinearRegression())
    print('before mdata.alg_logistic_regression')
    mdata.set_algorithm_obj(mdata.alg_logistic_regression, AlgLogisticRegression())
    print('before mdata.alg_decision_tree')
    mdata.set_algorithm_obj(mdata.alg_decision_tree, AlgDecisionTree())
    print('before mdata.alg_random_forest')
    mdata.set_algorithm_obj(mdata.alg_random_forest, AlgRandomForest())
    print('before mdata.alg_neural_net')
    mdata.set_algorithm_obj(mdata.alg_neural_net, AlgNeuralNet())
    print('before mdata.alg_svm')
    mdata.set_algorithm_obj(mdata.alg_svm, AlgSVM())
    print('before mdata.alg_gaussian_nb')
    mdata.set_algorithm_obj(mdata.alg_gaussian_nb, AlgGaussianNB())
    print('before mdata.alg_ensemble_stacking')
    mdata.set_algorithm_obj(mdata.alg_ensemble_stacking,
                            AlgEnsemble(mdata.alg_ensemble_stacking))
    print('before mdata.alg_ensemble_bagging')
    mdata.set_algorithm_obj(mdata.alg_ensemble_bagging,
                            AlgEnsemble(mdata.alg_ensemble_bagging))
    print('before mdata.alg_ensemble_adaboost')
    mdata.set_algorithm_obj(mdata.alg_ensemble_adaboost,
                            AlgEnsemble(mdata.alg_ensemble_adaboost))

    print('before housem.MainWindow')
    window = housem.MainWindow()
    print('before window.show()')
    window.show()
    window.raise_()

    app.exec_()
