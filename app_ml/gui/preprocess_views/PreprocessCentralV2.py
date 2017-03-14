import sys
import os
import logging as log
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import common.strings as strs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from preprocess_data import preprocess as prep

from gui.preprocess_views.PreprocessSettingV2 import PreprocessSettingViewV2

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

survived_color = '#6699ff'
died_color = '#ff6666'

class PreprocessViewV2(QWidget):
    def __init__(self, main_view=None, parent=None):
        super(PreprocessViewV2, self).__init__(parent)
        self.main_view = main_view
        self.setting_view = PreprocessSettingViewV2(self.main_view)
        self.X = None
        self.y = None
        self.idx_survived = None
        self.idx_died = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.hello_btn = QPushButton(strs.btn_hello)
        layout.addWidget(self.hello_btn)

        self.column_group = QButtonGroup()

    def get_settingview(self):
        return self.setting_view

    def showEvent(self, event):
        log.debug('start')
        self.init_columns_info()

    def init_columns_info(self):
        log.debug('start')
        col_names = prep.get_col_names()
        for col in col_names:
            print(col)

class ColumnPropertyLineView(QWidget):
    def __init__(self, col_name, missing_num, value_cat, btn_group, parent=None):
        super(ColumnPropertyLineView, self).__init__(parent)

        self.col_name = col_name

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.sel_radio = QRadioButton()
        self.sel_radio.clicked.connect(self.on_clicked_sel_radio)
        btn_group.addButton(self.sel_radio)
        layout.addWidget(self.sel_radio)

        self.name_label = QLabel(str(missing_num))
        layout.addWidget(self.name_label)

        self.missing_label = QLabel(str(self.missing_num))
        layout.addWidget(self.missing_label)

        self.cat_label = QLabel(value_cat)
        layout.addWidget(self.cat_label)

    def on_clicked_sel_radio(self):
        log.debug('start')



