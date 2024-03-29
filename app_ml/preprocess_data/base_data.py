# -*- coding: utf-8 -*-

import sys
import logging as log
import common.strings as strs
import numpy as np
import common.config as const
from common import utils
from common import config
import pandas as pd
from six.moves import cPickle as pickle
from collections import OrderedDict
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

log.basicConfig(format=strs.log_format,level=log.DEBUG,stream=sys.stderr)
np.set_printoptions(linewidth=1000)

from enum import Enum
class State(Enum):
    Init = 0
    Load = 1
    Preprocess = 2
    Train = 3

class BaseData(object):
    """
    BaseData는 데이터를 로딩하고 프리프로세싱 처리를 하는 클래스
    """
    def __init__(self, name):
        """
        BaseData constructor
        :param name: 데이터 이름
        :return: void
        """
        log.debug('start')
        self.data_name = name
        self.data_file = None
        self.column_names = None
        self.label_name = None
        self.X = None
        self.y = None
        self.org_X = None # deprecated
        self.org_y = None # deprecated
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.loaded_data = None
        self.null_indices = {}
        self.preprocessed = False
        self.cur_dimen_reduct_method = const.param_none
        self.cur_state = State.Init
        self.column_infos = OrderedDict()

        self.load_config()

    def set_columns(self, columns):
        self.column_names = columns

    def set_column_names(self, columns):
        log.debug('columns:%s' % columns)
        self.set_columns(columns)

    def get_column_names(self):
        return self.column_names

    def set_label_name(self, label):
        log.debug('label:%s' % label)
        self.label_name = label
        self.save_config()

    def get_label_name(self):
        return self.label_name

    def view_head(self):
        log.debug('start')
        with open(self.data_file) as f:
            for i in range(1, 6):
                print("LINE %d=== \n%s" % (i, f.readline()))
        log.debug('end')

    def load_data2(self):
        # hj-comment: 대용량 파일을 읽어올 때 어떻게 할지 고민을 해야함
        self.loaded_data = pd.read_csv(self.data_file)
        self.cur_state = State.Load
        self.column_names = self.loaded_data.columns

    def get_data(self):
        return self.loaded_data

    def get_cur_state(self):
        log.debug('Current State:%s', self.cur_state)
        return self.cur_state

    def _check_basic_data(self):
        if self.column_names is None:
            log.debug("ERROR:%s" % strs.error_columns_not_defined)
            return False

        if self.loaded_data  is None:
            log.debug("ERROR:%s" % strs.error_no_data)
            return False

        return True

    def analyze(self):
        log.debug('start')
        if self.column_names is None:
            log.debug(strs.error_columns_not_defined)
            return

        for col in self.column_names:
            # log.debug('{0} - {1}'.format(col, self.loaded_data[col]))
            colexist = col in self.column_infos.keys()
            if colexist is False:
                self.column_infos[col] = self._analyze_column(col)
            elif self.column_infos[col][strs.col_analyzed] is False:
                self.column_infos[col] = self._analyze_column(col)

        # utils.save_dict_csv(config.file_name_column_info, self.column_infos)
        self.save_config()

    def _analyze_column(self, col):
        info = OrderedDict()
        info[strs.col_name] = col
        org_col_data = self.loaded_data[col]

        # set default values
        BaseData.set_col_data_info(info, strs.col_use_value, True)
        BaseData.set_col_data_info(info, strs.col_analyzed, True)
        BaseData.set_col_data_info(info, strs.col_recommend_preprocess, '')

        # categorical data에 대한 range 세팅
        try:
            col_map = np.unique(self.loaded_data[col])
            if len(col_map) <= 10:
                log.debug('col_map:%s' % col_map)
                BaseData.set_col_data_info(info, strs.col_data_range, strs.col_data_range_small)
            else:
                log.debug('col_map length:%d' % len(col_map))
                BaseData.set_col_data_info(info, strs.col_data_range, strs.col_data_range_big)
        except TypeError:
            BaseData.set_col_data_info(info, strs.col_data_range, strs.col_data_range_none)

        # check missing data
        na_sum = org_col_data.isnull().sum()
        na_indices = self.loaded_data.index[org_col_data.isnull()]
        BaseData.set_col_data_info(info, strs.col_missing_count, na_sum)
        BaseData.set_col_data_info(info, strs.col_missing_indices, na_indices.values)

        log.debug('%s(%d): %d' % (col, len(self.loaded_data.index), na_sum))
        # log.debug('----- %s:%s' % (col, na_indices))

        # 데이터가 0인 개수
        zero_sum = 0
        if org_col_data.dtype == np.int:
            zero_sum = org_col_data[org_col_data == 0].count()
            BaseData.set_col_data_info(info, strs.col_data_type, strs.col_data_type_int)
        elif org_col_data.dtype == np.float or org_col_data.dtype == np.double:
            zero_sum = org_col_data[org_col_data == 0.0].count()
            BaseData.set_col_data_info(info, strs.col_data_type, strs.col_data_type_double)
        else:
            zero_sum = 0
            BaseData.set_col_data_info(info, strs.col_data_type, strs.col_data_type_str)
        BaseData.set_col_data_info(info, strs.col_zero_sum, zero_sum)

        # log.debug(BaseData.print_col_info(info))
        return info

    @staticmethod
    def set_col_data_info(info, key, value):
        info.setdefault(key, None)
        info[key] = value

    @staticmethod
    def print_col_info(info):
        log.debug('start')
        print('column name: ', info[strs.col_name])
        print('missing count: ', info[strs.col_missing_count])
        print('zero value count: ', info[strs.col_zero_sum])

    @staticmethod
    def desc(info):
        log.debug('start')
        # if self._check_basic_data() is False:
        #     return
        #
        # # check missing data
        # for col in self.column_names:
        #     na_sum = self.loaded_data[col].isnull().sum()
        #     na_indices = self.loaded_data.index[self.loaded_data[col].isnull()]
        #     log.debug('%s(%d): %d' % (col, len(self.loaded_data.index), na_sum))
        #     if na_indices.size > 0:
        #         log.debug('----- %s:%s' % (col, na_indices))
        #     self.null_indices.setdefault(col, [])
        #     self.null_indices[col] = na_indices

    def get_col_infos(self):
        log.debug('start')
        if len(self.column_infos) == 0:
            self.analyze()

        return self.column_infos

    def get_col_info(self, col_name):
        log.debug('start')
        self.get_col_infos()
        return self.column_infos[col_name]

    def get_col_values(self, col_name):
        return self.loaded_data[col_name]

    def get_label_values(self):
        if self.label_name is not None and self.label_name != '':
            return self.loaded_data[self.label_name]
        else:
            return None

    def preprocess(self):
        log.debug('start')

    def get_config_file_name(self):
        file_name = config.file_name_config + '_' + self.data_name + '.pkl'
        log.debug(file_name)
        return file_name

    def set_column_config(self, col_info):
        col_name = col_info[strs.col_name]
        self.column_infos[col_name] = col_info


    def load_config(self):
        log.debug('start')
        try:
            with open(self.get_config_file_name(), 'rb') as f:
                config = pickle.load(f)
                self.data_name = config['data_name']
                self.label_name = config['label_name']
                self.column_infos = config['column_infos']
                log.debug(self.column_infos)
        except Exception as e:
            log.debug('Unable to read data from {0}:{1}'.format(self.get_config_file_name(), e))

    def save_config(self):
        log.debug('start')
        try:
            with open(self.get_config_file_name(), 'wb') as f:
                config = {
                    'data_name': self.data_name,
                    'label_name': self.label_name,
                    'column_infos': self.column_infos,
                }
                pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            log.debug('Unable to save data to {0}:{1}'.format(self.get_config_file_name(), e))

    def preprocess_data(self, force_process = False):
        """
        프리프로세싱을 진행한다. 데이터가 로드되지 않았다면 데이터로드부터 한다. 데이터 로딩후에 다음과 같은 프리프로세싱을 진행한다.
        1) 미싱데이터 처리
        2) One Hot Encoding
        3) Standardization
        4) Outlier 처리 
        :return: void
        """
        # if self.cur_dimen_reduct_method != mdata.get_dimem_reduct_method():
        #     self.preprocessed = False
        log.debug('>>>>> start')

        self.desc_data()
        if self.preprocessed == True and force_process == False:
            return

        # 레이블이 지정되지 않았으면 메시지 띄우고 리턴
        if self.label_name is None:
            log.debug(strs.error_no_label_value)
            return

        # dataset, label 구분
        self.y = self.loaded_data[self.label_name]
        self.X = self.loaded_data.ix[:, self.loaded_data.columns != self.label_name]
        log.debug("X dataset:{}".format(self.X.columns))

        # 프리프로세싱 시작
        self.process_missing_data()

        # 다음의 함수들을 새로운 프로세스에 맞게 수정해야함
        # self.merge_data(self.X) # hj-next
        pd.to_pickle(self.X, './used_X1.pkl')
        self.convert_categorical_data(self.X) # manipulate categorical data, convert_data_type
        pd.to_pickle(self.X, './used_X2.pkl')
        # self.X = self.convert_to_dummy_data(self.X)
        # log.debug('X first row: {0}'.format(self.X.iloc[0]))
        used_columns = [col[strs.col_name] for col in self.column_infos.values() if col[strs.col_use_value] is True]
        log.debug('Used Colunms: {0}'.format(used_columns))
        self.X = self.X[used_columns]
        pd.to_pickle(self.X, './used_X3.pkl')
        log.debug('X used for first row: {0}'.format(self.X.iloc[0]))
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)
        self.X_df = self.X
        self.X = self.X.values
        self.y = self.y.values
        # y = one_hot_encode(y)
        # self.X = self.standardize_data(self.X)  # standardize
        log.debug('>>>>> Processed Data:\n{0}'.format(self.X[:5]))

        # split data into train, validation, test data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.20, random_state=0)
        self.X_train, self.X_valid, self.y_train, self.y_valid = \
            train_test_split(self.X_train, self.y_train, test_size=0.20, random_state=0)

        log.debug('X shape: %s, y shape: %s' % (self.X.shape, self.y.shape))
        log.debug('X train shape: %s, y train shape: %s' % (self.X_train.shape, self.y_train.shape))
        log.debug('X valid shape: %s, y valid shape: %s' % (self.X_valid.shape, self.y_valid.shape))
        log.debug('X test shape: %s, y test shape: %s' % (self.X_test.shape, self.y_test.shape))
        self.preprocessed = True

        log.debug('##### data name: %s' % self.data_name)
        return

    def desc_data(self): # hj-next
        pass
        # for col_info in self.column_infos:
        #     log.debug('column info: {0}'.format(col_info))
        #     log.debug('{0}: {1}'.format(col_info[strs.col_name], col_info[strs.col_use_value]))

    def process_missing_data(self):
        log.debug('start')
        self.analyze()

        self.check_missing_data()

        if 'Age' in self.column_names:
            log.debug("process missing data for Age")
            # self.X.loc[:, 'Age'] = self.X["Age"].fillna(self.X["Age"].median())
            age_medain = self.X["Age"].median()
            self.X["Age"].fillna(age_medain, inplace=True)

        if 'Embarked' in self.column_names:
            log.debug("process missing data for Embarked")
            # self.X.loc[:, 'Embarked'] = self.X['Embarked'].fillna('S')
            self.X['Embarked'].fillna('S', inplace=True)

        if 'Fare' in self.column_names and 'Pclass' in self.column_names:
            log.debug("process missing data for Fare")
            # nullfares = X[X.Fare == 0]
            nullfares = self.X[(self.X.Fare == 0) | (self.X.Fare.isnull())]
            log.debug('len of nullfares:{0}'.format(nullfares))
            for index in nullfares.index:
                clsFare = self.X[self.X.Pclass == self.X.loc[index, 'Pclass']][self.X.Fare != 0].Fare.mean()
                log.debug("Pclass: %s, Fare: %f" % (self.X.loc[index, 'Pclass'], clsFare))
                self.X.loc[index, 'Fare'] = clsFare
            log.debug('{0}'.format(self.X.Fare))

        self.check_missing_data()

    def check_missing_data(self):
        for col in self.X.columns:
            dtype = self.X[col].dtype
            if dtype == np.int:
                log.debug("{0}(int): # of null: {1}, # of zero: {2}".format(
                              col, self.X[col].isnull().sum(), (self.X[col] == 0.0).sum()))
            elif dtype == np.float:
                log.debug("{0}(float): # of null: {1}, # of zero: {2}".format(
                              col, self.X[col].isnull().sum(), (self.X[col] == 0.0).sum()))
            else:
                log.debug("{0}(object): # of null: {1}".format(col, self.X[col].isnull().sum()))

    def convert_categorical_data(self, X):
        for col_info in self.column_infos.values():
            log.debug('{0}, type: {1}'.format(col_info[strs.col_name], col_info[strs.col_data_type]))
            # log.debug('col name: ({0})'.format(col_info[strs.col_name]))
            if col_info[strs.col_use_value]:
                log.debug("{0}: col_data_range: {1}".format(col_info[strs.col_name], col_info[strs.col_data_range]))
                if col_info[strs.col_name] == 'Sex':
                    log.debug("convert Sex data through LabelBinarizer")
                    encoder = LabelBinarizer()
                    X.loc[:, 'Sex'] = encoder.fit_transform(X['Sex'])
                elif col_info[strs.col_name] == 'Pclass':
                    log.debug('convert Pclass data through OneHotEncoder')
                    encoder = OneHotEncoder()
                    X.Pclass = encoder.fit_transform(X.Pclass)
                elif col_info[strs.col_name] == 'Embarked':
                    log.debug('convert Embarked data through OneHotEncoder')
                    embarked_encoded, categories = X.Embarked.factorize()
                    encoder = OneHotEncoder()
                    X.Embarked = encoder.fit_transform(embarked_encoded.reshape(-1, 1))


    ###########################################################################
    # hj-deprecated
    ###########################################################################


    def split_data(self):
        log.debug('not implemented')

    def get_titanic(self):
        return self.loaded_data

    def get_X_train(self, include_valid=False, size=-1):
        log.debug('>>>>> start - shape of X_train: %s' % (self.X_train.shape,))
        log.debug('##### data name: %s' % self.data_name)
        if include_valid is True and self.X_valid is not None:
            temp_X_train = np.vstack((self.X_train, self.X_valid))
            log.debug('temp_X_train shape: %s' % (temp_X_train.shape,))
            dataset = temp_X_train
        else:
            dataset = self.X_train

        if size != -1:
            return dataset[:size]
        else:
            return dataset

    def get_y_train(self, include_valid=False, one_hot_encoding=False, size=-1):
        log.debug('##### data name: %s' % self.data_name)
        if include_valid is True and self.y_valid is not None:
            if len(self.y_train.shape) == 2:
                temp_y_train = np.vstack((self.y_train, self.y_valid))
            else:
                temp_y_train = np.concatenate((self.y_train, self.y_valid))
        else:
            temp_y_train = self.y_train

        if one_hot_encoding is False:
            labels = self._convert_values_no_ohe(temp_y_train)
        else:
            labels = self.y_train

        if size != -1:
            return labels[:size]
        else:
            return labels

    def get_X_valid(self, size=-1):
        log.debug('##### data name: %s' % self.data_name)
        if size != -1:
            return self.X_valid[:size]
        else:
            return self.X_valid

    def get_y_valid(self, one_hot_encoding=False, size=-1):
        log.debug('##### data name: %s' % self.data_name)
        if one_hot_encoding is False:
            labels = self._convert_values_no_ohe(self.y_valid)
        else:
            labels = self.y_valid

        if size != -1:
            return labels[:size]
        else:
            return labels

    def get_X_test(self, size=-1):
        log.debug('##### data name: %s' % self.data_name)
        if size != -1:
            return self.X_test[:size]
        else:
            return self.X_test

    def get_y_test(self, one_hot_encoding=False, size=-1):
        log.debug('##### data name: %s' % self.data_name)
        if one_hot_encoding is False:
            labels = self._convert_values_no_ohe(self.y_test)
        else:
            labels = self.y_test

        if size != -1:
            return labels[:size]
        else:
            return labels

    def get_x_field_count(self):
        log.debug('start - shape of X_train: %s' % (self.X_train.shape,))
        log.debug('##### data name: %s' % self.data_name)
        if self.X_train is None:
            log.debug(strs.error_obj_null)
            return None
        return self.X_train.shape[1]

    def get_y_value_count(self):
        log.debug('start - shape of y_train: %s' % (self.y_train.shape,))
        log.debug('##### data name: %s' % self.data_name)
        if self.y_train is None:
            log.debug(strs.error_obj_null)
            return None

        if len(self.y_train.shape) == 1:  # binary classification
            return 1
        else:
            return self.y_train.shape[1]

    def get_batch_size(self):
        log.debug('>>>>> start')
        log.debug('##### data name: %s' % self.data_name)
        return 0

    def get_hidden_size(self):
        log.debug('>>>>> start')
        log.debug('##### data name: %s' % self.data_name)
        return 0

    def get_step_size(self):
        log.debug('>>>>> start')
        log.debug('##### data name: %s' % self.data_name)
        return 0

    def load_train_data(self):
        self.loaded_data = pd.read_csv(self.data_file)
        log.debug('##### specified data columns: %s' % self.column_names)
        X = self.loaded_data[self.column_names]
        y = self.loaded_data['Survived']
        log.debug('test data: \n{0}'.format(X.head(10)))
        return X, y

    def _convert_values_no_ohe(self, values):
        log.debug('##### data name: %s' % self.data_name)
        if self.get_y_value_count() == 1:
            return values
        else:
            return np.argmax(values, axis=1)

    def load_data(self):
        log.debug('start')