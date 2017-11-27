import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import logging as log
import common.strings as strs
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from preprocess_data.base_data import BaseData

log.basicConfig(format=strs.log_format,level=log.DEBUG,stream=sys.stderr)

# hj-comment: move load_data, process_missing_data, convert_to_numeric functions to parent class


class KaggleData2(BaseData):
    def __init__(self, name):
        log.debug('start')
        super(KaggleData2, self).__init__(name)
        self.data_file = './data/train.csv'
        self.column_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.column_names = None

    def process_missing_data(self):
        log.debug('start')
        super(KaggleData2, self).process_missing_data()

        if 'Age' in self.column_names:
            log.debug("process missing data for Age")
            self.X.loc[:, 'Age'] = self.X["Age"].fillna(self.X["Age"].median())

        if 'Embarked' in self.column_names:
            log.debug("process missing data for Embarked")
            self.X.loc[:, 'Embarked'] = self.X['Embarked'].fillna('S')

        if 'Fare' in self.column_names and 'Pclass' in self.column_names:
            log.debug("process missing data for Fare")
            # nullfares = X[X.Fare == 0]
            nullfares = self.X[(self.X.Fare == 0) | (self.X.Fare.isnull())]
            log.debug('len of nullfares:{0}'.format(nullfares))
            for index in nullfares.index:
                clsFare = self.X[self.X.Pclass == self.X.loc[index, 'Pclass']][self.X.Fare != 0].Fare.mean()
                # log.debug("Pclass: %s, Fare: %f" % (X.loc[index, 'Pclass'], clsFare))
                self.X.loc[index, 'Fare'] = clsFare