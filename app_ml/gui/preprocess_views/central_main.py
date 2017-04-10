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
from gui.preprocess_views.setting_main import PreprocessSettingViewV2
from gui.preprocess_views.central_detail import ColumnDetailView
from gui.preprocess_views.central_columns import ColumnsView

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

class CentralView(QWidget):
    def __init__(self, main_view=None, parent=None):
        super(CentralView, self).__init__(parent)
        self.setting_view = PreprocessSettingViewV2(main_view)

        # 최상위 레이아웃과 스크롤 영역 세팅
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # 컬럼들의 목록을 보여주는 커스텀 위젯
        self.cols_view = ColumnsView(self, self)
        self.layout.addWidget(self.cols_view)

        # 한 컬럼 데이터에 대한 세부 정보 커스텀 위젯
        self.detail_view = ColumnDetailView(self)
        self.detail_view_layout = QVBoxLayout()
        self.detail_view.setLayout(self.detail_view_layout)
        self.detail_view.setMinimumHeight(config.preprocess_central_col_detail_height)
        self.layout.addWidget(self.detail_view)

    def get_settingview(self):
        return self.setting_view

    def reset_detail_view(self):
        log.debug('>>>>>> start')
        self.detail_view.reset();

    def show_detail_info(self, colname):
        self.detail_view.show_detail_info(colname)

    def backup(self):
        pass
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.blue);
        # self.detail_view.setAutoFillBackground(True);
        # self.detail_view.setPalette(pal);


