# -*- coding: utf-8 -*-
import sys
import os
import logging as log
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import common.strings as strs
from common import config
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from preprocess_data import preprocess as prep
from matplotlib.figure import Figure
from common.utils_ui import *

from gui.preprocess_views.setting_main import PreprocessSettingViewV2

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

survived_color = '#6699ff'
died_color = '#ff6666'

# hj-comment, histogram 관련해서 구현시 참고: https://docs.google.com/document/d/1VT-QFtqGPK16nqk6RPSLs6zMEGQ742askoL2gjdcTXo/edit


class CentralChart(QWidget):
    def __init__(self, col_name, parent=None):
        super(CentralChart, self).__init__(parent)
        log.debug('start')

        if col_name is None or col_name == '':
            log.debug(strs.error_column_name_not_specified)

        # data
        self.col_name = col_name
        self.col_values = None
        self.label_values = None

        # layout
        layout = QGridLayout()
        self.setLayout(layout)

        # widget - Missing value counts
        wg_missing, self.missing_cnts_input = input_widget_horizontal(strs.ctl_txt_column_detail_info_missing)
        layout.addWidget(wg_missing, 0, 0)
        layout.setColumnStretch(0, 2)

        # widget - Value type
        wg_type, self.value_type_input = input_widget_horizontal(strs.ctl_txt_column_detail_info_type)
        layout.addWidget(wg_type, 0, 1)
        layout.setColumnStretch(1, 2)

        # widget - Value set
        wg_type, self.value_set_input = input_widget_vertical(strs.ctl_txt_column_detail_info_value_set,
                                                              config.default_view_height,
                                                              config.column_detail_info_cont_view_height,
                                                              config.column_detail_info_cont_view_max_width)
        layout.addWidget(wg_type, 1, 0, 1, 4)

        # widget - Description
        wg_type, self.desc_input = input_widget_vertical(strs.ctl_txt_column_detail_info_desc,
                                                         config.default_view_height,
                                                         config.column_detail_info_cont_view_height,
                                                         config.column_detail_info_cont_view_max_width)
        layout.addWidget(wg_type, 2, 0, 1, 4)

    # Show Descriptive Information
    def show_desc_info(self, col_name):
        log.debug('>>>>>> column name: {0}'.format(col_name))
        self.col_name = col_name
        self.col_values = prep.get_col_values(self.col_name)
        self.label_values = prep.get_label_values()

        col_info = prep.get_col_info(self.col_name)
        self.missing_cnts_input.setText(str(col_info[strs.col_missing_count]))
        self.value_set_input.setText(str(np.unique(self.col_values)))
        self.value_type_input.setText(col_info[strs.col_data_type])


