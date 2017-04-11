# -*- coding: utf-8 -*-
import logging as log
import sys

from PyQt5.QtWidgets import *
from gui.preprocess_views.column_info.column_main_view import ColumnMainView

import common.strings as strs
from common import config
from gui.preprocess_views.column_info.detail_view import DetailView
from gui.preprocess_views.setting_main import PreprocessSettingViewV2
from gui.preprocess_views.column_info.column_main_view import ColumnMainView

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

class HeatmapView(QWidget):
    def __init__(self, main_view=None, parent=None):
        super(HeatmapView, self).__init__(parent)
        self.setting_view = PreprocessSettingViewV2(main_view)
        self.layout = QVBoxLayout()

        testbtn = QPushButton('hello')
        self.layout.addWidget(testbtn);

        self.setLayout(self.layout)

    def get_settingview(self):
        return self.setting_view