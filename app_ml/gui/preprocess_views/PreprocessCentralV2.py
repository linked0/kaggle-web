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

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.col_cont = QWidget()
        self.col_cont_layout = QGridLayout()
        self.col_cont.setLayout(self.col_cont_layout)
        self.col_cont.setMaximumHeight(config.preprocess_central_col_cont_height)
        self.layout.addWidget(self.col_cont)
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.red);
        # self.col_cont.setAutoFillBackground(True);
        # self.col_cont.setPalette(pal);

        self.detail_cont = QWidget()
        self.detail_cont_layout = QVBoxLayout()
        self.detail_cont.setLayout(self.detail_cont_layout)
        self.layout.addWidget(self.detail_cont)

        self.col_group = QButtonGroup()

    def get_settingview(self):
        return self.setting_view

    def showEvent(self, event):
        log.debug('start')
        infos = prep.get_col_infos()
        if infos is not None:
            row = 0
            for info in infos:
                self.col_cont_layout.addWidget(ColumnPropertyView(info, self.col_group), row, 0)
                row += 1


class ColumnPropertyView(QWidget):
    def __init__(self, col_info, btn_group, parent=None):
        super(ColumnPropertyView, self).__init__(parent)

        self.col_name = col_info['name']

        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setMinimumHeight(config.preprocess_central_item_height)

        self.sel_radio = QRadioButton()
        self.sel_radio.clicked.connect(self.on_clicked_sel_radio)
        btn_group.addButton(self.sel_radio)
        layout.addWidget(self.sel_radio)

        self.name_label = QLabel(str(col_info['name']))
        layout.addWidget(self.name_label)
        pal = QPalette(); # background 지정
        pal.setColor(QPalette.Background, Qt.red);
        self.name_label.setAutoFillBackground(True);
        self.name_label.setPalette(pal);

        self.missing_label = QLabel(str(col_info['missing_num']))
        layout.addWidget(self.missing_label)

        # background 지정
        pal = QPalette();
        pal.setColor(QPalette.Background, Qt.cyan);
        self.setAutoFillBackground(True);
        self.setPalette(pal);

        log.debug('Contents Margins:%s' % (self.getContentsMargins(),))

        # self.cat_label = QLabel(value_cat)
        # layout.addWidget(self.cat_label)

    def on_clicked_sel_radio(self):
        log.debug('start')



