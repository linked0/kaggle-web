# -*- coding: utf-8 -*-
import logging as log
import sys

from PyQt5.QtWidgets import *
from gui.preprocess_views.column_info.column_main_view import ColumnMainView

import common.strings as strs
from common import config
from gui.preprocess_views.setting_main import PreprocessSettingViewV2
from gui.preprocess_views.column_info.column_main_view import ColumnMainView
from gui.preprocess_views.heat_map.heatmap_view import HeatmapView

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

class PreprocessCentralView(QWidget):
    def __init__(self, main_view=None, parent=None):
        super(PreprocessCentralView, self).__init__(parent)
        self.setting_view = PreprocessSettingViewV2(main_view)
        self.layout = QVBoxLayout()
        self.tabWidget = QTabWidget()

        # 컬럼 정보 탭
        self.curscore = ColumnMainView()
        self.tabWidget.addTab(self.curscore, strs.preprocess_columns_view)

        # 히트맵 탭
        self.bestscore = HeatmapView()
        self.tabWidget.addTab(self.bestscore, strs.preprocess_heatmap_view)

        self.layout.addWidget(self.tabWidget)
        self.setLayout(self.layout)

    def get_settingview(self):
        return self.setting_view


