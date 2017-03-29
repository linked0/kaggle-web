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
from gui.preprocess_views.PreprocessSettingV2 import PreprocessSettingViewV2
from gui.preprocess_views.ColumnDetail import ColumnDetailView

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

class PreprocessViewV2(QWidget):
    def __init__(self, main_view=None, parent=None):
        super(PreprocessViewV2, self).__init__(parent)
        self.main_view = main_view
        self.setting_view = PreprocessSettingViewV2(self.main_view)
        self.X = None
        self.y = None
        self.idx_survived = None
        self.idx_died = None
        self.col_info_set = False   # 컬럼 정보가 세팅되었는지 확인
        self.prev_label_name = ""

        # 최상위 레이아웃과 스크롤 영역 세팅
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # Data Label 세팅 화면
        lable_sel_text = QLabel(strs.preprocess_central_label_sel_title)
        self.layout.addWidget(lable_sel_text)
        self.label_sel_combo = QComboBox()
        self.label_sel_combo.currentTextChanged.connect(self.on_combo_changed)
        self.layout.addWidget(self.label_sel_combo)

        # 컬럼 목록 영역: 스크롤 영역과 그 안에 포함되는 컬럼들 정보 위젯
        cols_sect_text = QLabel(strs.preprocess_central_col_sect_title)
        self.layout.addWidget(cols_sect_text)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(config.preprocess_central_col_cont_height)
        self.layout.addWidget(scroll)

        # scroll(QScrollArea)에 포함될 layout, 나중에 이 layout에 컬럼 목록이 들어가게됨
        self.cols_view = QWidget()
        # self.cols_view.setStyleSheet("border:1px solid rgb(0, 0, 0); ")
        self.cols_view_layout = QGridLayout()
        self.cols_view.setLayout(self.cols_view_layout)
        scroll.setWidget(self.cols_view)
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.red);
        # self.cols_view.setAutoFillBackground(True);
        # self.cols_view.setPalette(pal);

        # 한 컬럼 데이터에 대한 세부 정보 위젯
        self.detail_view = ColumnDetailView(self)
        self.detail_view_layout = QVBoxLayout()
        self.detail_view.setLayout(self.detail_view_layout)
        self.detail_view.setMinimumHeight(config.preprocess_central_col_detail_height)
        self.layout.addWidget(self.detail_view)
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.blue);
        # self.detail_view.setAutoFillBackground(True);
        # self.detail_view.setPalette(pal);

        self.col_group = QButtonGroup()

    def get_settingview(self):
        return self.setting_view

    def on_combo_changed(self, seltext):
        if prep.get_label_name() == self.prev_label_name:
            return

        log.debug('changed label: ' + seltext)
        if seltext != strs.btn_select_one:
            self.prev_label_name = prep.get_label_name()
            prep.set_data_label(seltext)
            self.reset_info()


    # 컬럼 이름들 보여주기
    def showEvent(self, event):
        log.debug('start')

        if self.col_info_set is True and prep.get_dirty_flag() is False:
            return

        infos = prep.get_col_infos()
        if infos is not None:
            row = 0
            col = 0
            self.label_sel_combo.addItem(strs.btn_select_one)

            for info in infos:
                # add to label selection combo
                self.label_sel_combo.addItem(info['name'])

                # add to columns info section
                prop_view = ColumnPropertyView(info, self.col_group, self.detail_view, self)
                self.cols_view_layout.addWidget(prop_view, row, col)
                self.cols_view_layout.setRowStretch(row, 0)
                if col == 4:
                    row += 1
                    col = 0
                else:
                    col += 1
            self.cols_view.setMaximumHeight((row+1) * (config.preprocess_central_item_height+5))
            self.col_info_set = True;

            log.debug('label_name:{0}'.format(prep.get_label_name()))
            if prep.get_label_name() is not None:
                index = self.label_sel_combo.findText(prep.get_label_name())
                if index >= 0:
                    self.label_sel_combo.setCurrentIndex(index)

                # 레이블 컬럼은 X값으로 선택되지 않도록 처리
                for idx in reversed(range(self.cols_view_layout.count())):
                    # takeAt does both the jobs of itemAt and removeWidget
                    # namely it removes an item and returns it
                    widget = self.cols_view_layout.itemAt(idx).widget()

                    if widget is not None:
                        # widget will be None if the item is a layout
                        widget.setLabelColumn()


    def reset_info(self):
        self.detail_view.reset();

        log.debug('prev label:%s, cur label:%s' % (self.prev_label_name, prep.get_label_name()))
        prev_widget = None
        cur_widget = None
        items = (self.cols_view_layout.itemAt(i).widget() for i in range(self.cols_view_layout.count()))
        for widget in items:
            if widget.col_name == self.prev_label_name:
                prev_widget = widget
            elif widget.col_name == prep.get_label_name():
                cur_widget = widget
        log.debug('selected cur widget:%s' % cur_widget)


class ColumnPropertyView(QWidget):
    def __init__(self, col_info, btn_group, detail_view, parent=None):
        super(ColumnPropertyView, self).__init__(parent)

        self.col_name = col_info['name']
        self.parent = parent
        self.detail_view = detail_view

        # border settng
        self.setStyleSheet("border:1px solid rgb(0, 0, 0); ")

        # layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setMinimumHeight(config.preprocess_central_item_height)

        self.sel_radio = QRadioButton()
        self.sel_radio.clicked.connect(self.on_clicked_sel_radio)
        self.sel_radio.setMaximumWidth(config.preprocess_central_radio_width)
        btn_group.addButton(self.sel_radio)
        layout.addWidget(self.sel_radio)

        self.name_label = QLabel(str(col_info['name']))
        layout.addWidget(self.name_label)

        # self.missing_label = QLabel(str(col_info['missing_num']))
        # layout.addWidget(self.missing_label)

        self.use_check = QCheckBox()
        self.use_check.setMaximumWidth(config.preprocess_central_radio_width)
        self.use_check.setChecked(col_info['use_value'])
        layout.addWidget(self.use_check)

        # background 지정
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.cyan);
        # self.setAutoFillBackground(True);
        # self.setPalette(pal);

        # self.cat_label = QLabel(value_cat)
        # layout.addWidget(self.cat_label)

    def on_clicked_sel_radio(self):
        self.detail_view.show_detail_info(self.col_name)

    def setLabelColumn(self):
        log.debug('start')
        self.use_check.setChecked(False)
        self.use_check.setCheckable(False)
