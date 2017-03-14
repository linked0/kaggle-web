import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sys
import logging as log
import common.strings as strs
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from preprocess_data.base_data import BaseData

log.basicConfig(format=strs.log_format,level=log.DEBUG,stream=sys.stderr)

# hj-comment: move load_data, process_missing_data, convert_to_numeric functions to parent class


class KaggleData2(BaseData):
    def __init__(self):
        super(KaggleData2, self).__init__()

        self.data_file = './data/train.csv'
        self.column_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.column_names = None
        log.debug('start')