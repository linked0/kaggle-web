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
from common.ui_utils import *

from gui.preprocess_views.PreprocessSettingV2 import PreprocessSettingViewV2

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

survived_color = '#6699ff'
died_color = '#ff6666'

class ColumnInfoView(QWidget):
    def __init__(self, col_name, parent=None):
        super(ColumnInfoView, self).__init__(parent)
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
        wg_missing = input_widget_horizontal(strs.ctl_txt_column_detail_info_missing)
        layout.addWidget(wg_missing, 0, 0, 2)

        # widgeet - Value type
        wg_type = input_widget_horizontal(strs.ctl_txt_column_detail_info_type)
        layout.addWidget(wg_type, 1, 2, 2)

        # widget - Value set
        wg_type = input_widget_vertical(strs.ctl_txt_column_detail_info_type,
                                        config.style_default_view_height,
                                        config.style_column_detail_info_cont_view_height)
        layout.addWidget(wg_type, 1, 0, 4)

        # widget - Description
        wg_type = input_widget_vertical(strs.ctl_txt_column_detail_info_desc,
                                        config.style_default_view_height,
                                        config.style_column_detail_info_cont_view_height)
        layout.addWidget(wg_type, 2, 0, 4)

    def show_info(self):
        self.init_data()


    def init_data(self):
        self.col_values = prep.get_col_values(self.col_name)
        self.label_values = prep.get_label_values()
