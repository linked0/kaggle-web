import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import logging as log
import common.strings as strs
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from preprocess_data.base_data import BaseData

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

data_root = './data/house_prices/'
train_data_file = data_root + 'train.csv'
test_data_file = data_root + 'test.csv'
data_desc_file = data_root + 'data_description.txt'

class KaggleHousePrices(BaseData):
    def __init__(self, name):
        super(KaggleHousePrices, self).__init__(name)
        self.data_file = train_data_file
        log.debug('start')

    def load_data(self):
        log.debug('features:{0}'.format(self.X.columns))