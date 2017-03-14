# -*- coding: utf-8 -*-

import sys
import logging as log
import common.strings as strs
import numpy as np
import common.config as const
import pandas as pd

log.basicConfig(format=strs.log_format,level=log.DEBUG,stream=sys.stderr)
np.set_printoptions(linewidth=1000)

from enum import Enum
class State(Enum):
    Init = 0
    Load = 1
    Preprocess = 2
    Train = 3

class BaseData(object):
    def __init__(self):
        log.debug('start')
        self.data_name = None
        self.data_file = None
        self.column_names = None
        self.label_name = None
        self.X = None
        self.y = None
        self.org_X = None
        self.org_y = None
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

    def init(self):
        self.data_name = None
        self.data_file = None
        self.column_names = None
        self.X = None
        self.y = None
        self.org_X = None
        self.org_y = None
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

    def set_data_file(self, path):
        self.init()
        log.debug('start')
        self.data_file = path

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

    def desc(self):
        log.debug('start')
        if self._check_basic_data() is False:
            return

        # check missing data
        for col in self.column_names:
            na_sum = self.loaded_data[col].isnull().sum()
            na_indices = self.loaded_data.index[self.loaded_data[col].isnull()]
            log.debug('%s(%d): %d' % (col, len(self.loaded_data.index), na_sum))
            if na_indices.size > 0:
                log.debug('----- %s:%s' % (col, na_indices))
            self.null_indices.setdefault(col, [])
            self.null_indices[col] = na_indices

    def split_data(self):
        log.debug('not implemented')

    def preprocess_data(self):
        # if self.cur_dimen_reduct_method != mdata.get_dimem_reduct_method():
        #     self.preprocessed = False
        log.debug('##### data name: %s' % self.data_name)
        return

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

    ###########################################################################
    # hj-deprecated
    ###########################################################################
    def load_data(self):
        log.debug('start')