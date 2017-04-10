# -*- coding: utf-8 -*-
import sys
import os
import logging as log
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import common.strings as strs
from common import config
import numpy as np
from common.utils_ui import *

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)


class ColumnProperty(QWidget):
    def __init__(self, col_info, btn_group, ctrl_view, parent=None):
        super(ColumnProperty, self).__init__(parent)

        self.col_name = col_info[strs.col_name]
        self.ctrl_view = ctrl_view
        self.parent = parent

        # border settng
        # self.setStyleSheet("border:1px solid rgb(0, 0, 0); ")

        # layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setMinimumHeight(config.preprocess_central_item_height)

        self.sel_radio = QRadioButton()
        self.sel_radio.clicked.connect(self.on_clicked_sel_radio)
        self.sel_radio.setMaximumWidth(config.preprocess_central_radio_width)
        btn_group.addButton(self.sel_radio)
        layout.addWidget(self.sel_radio)

        self.name_label = QLineEdit(str(col_info[strs.col_name]))
        self.name_label.setReadOnly(True)
        layout.addWidget(self.name_label)

        self.use_check = QCheckBox()
        self.use_check.setMaximumWidth(config.preprocess_central_radio_width)
        self.use_check.setChecked(col_info[strs.col_use_value])
        layout.addWidget(self.use_check)

        # background 지정
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.cyan);
        # self.setAutoFillBackground(True);
        # self.setPalette(pal);

        # self.cat_label = QLabel(value_cat)
        # layout.addWidget(self.cat_label)

    def on_clicked_sel_radio(self):
        self.ctrl_view.show_detail_info(self.col_name)
        pass

    def setLabelColumn(self):
        log.debug('start')
        self.use_check.setChecked(False)
        self.use_check.setCheckable(False)