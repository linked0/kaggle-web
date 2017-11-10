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
from preprocess_data import preprocess as prep

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)


class PropertyControl(QWidget):
    def __init__(self, col_info, btn_group, ctrl_view, parent=None):
        super(PropertyControl, self).__init__(parent)

        log.debug('col info: %s' % col_info)
        log.debug('id of col info({0}): {1}'.format(col_info[strs.col_name], id(col_info)))
        self.col_info = col_info
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
        log.debug('use check: %s' % col_info[strs.col_use_value])
        self.use_check.setChecked(col_info[strs.col_use_value])
        self.use_check.stateChanged.connect(self.on_clicked_use_check)
        layout.addWidget(self.use_check)

        # background 지정
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.cyan);
        # self.setAutoFillBackground(True);
        # self.setPalette(pal);

        # self.cat_label = QLabel(value_cat)
        # layout.addWidget(self.cat_label)

    def on_clicked_sel_radio(self):
        self.ctrl_view.show_detail_info(self.col_name, self)
        pass

    def on_clicked_use_check(self):
        log.debug('start')
        self.col_info[strs.col_use_value] = self.use_check.isChecked()
        print(">>>>>> on_clicked_use_check: config:{0}".format(self.col_info))
        prep.save_data_store_config()
        pass

    def setLabelColumn(self):
        log.debug('start')

    def exclude_column(self, col_name):
        self.use_check.setChecked(False)
        log.debug('start: %s' % col_name)