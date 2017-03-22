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

        # 스크롤 영역과 그 안에 포함되는 컬럼들 정보 위젯
        cols_sect_text = QLabel(strs.preprocess_central_col_sect_title)
        self.layout.addWidget(cols_sect_text)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(config.preprocess_central_col_detail_height)
        self.layout.addWidget(scroll)

        # create Columns View
        self.cols_view = QWidget()
        # self.cols_view.setStyleSheet("border:1px solid rgb(0, 0, 0); ")
        self.cols_view_layout = QGridLayout()
        self.cols_view.setLayout(self.cols_view_layout)
        scroll.setWidget(self.cols_view)
        # pal = QPalette();
        # pal.setColor(QPalette.Background, Qt.red);
        # self.cols_view.setAutoFillBackground(True);
        # self.cols_view.setPalette(pal);

        # 단일 컬럼 세부 정보 위젯
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
                if col == 2:
                    row += 1
                    col = 0
                else:
                    col += 1
            self.cols_view.setMaximumHeight(row * (config.preprocess_central_item_height + 5))
            self.col_info_set = True;

            log.debug('label_name:{0}'.format(prep.get_label_name()))
            if prep.get_label_name() != None:
                index = self.label_sel_combo.findText(prep.get_label_name())
                if index >= 0:
                    self.label_sel_combo.setCurrentIndex(index)

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

        self.missing_label = QLabel(str(col_info['missing_num']))
        layout.addWidget(self.missing_label)

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


class ColumnDetailView(QWidget):
    def __init__(self, parent=None):
        super(ColumnDetailView, self).__init__(parent)
        log.debug('start')

        # data
        self.cur_col = None
        self.col_values = None

        # layout
        layout = QHBoxLayout()
        self.setLayout(layout)

        # distribution plotting view
        self.fig, self.axes = plt.subplots(1, 2)
        self.hist_plot = self.axes[0]
        self.chart_plot = self.axes[1]
        self.plot_canvas = FigureCanvas(self.fig)
        layout.addWidget(self.plot_canvas)

        log.debug('parent: %s', self.parent)

    def show_detail_info(self, col_name):
        log.debug('start - %s' % col_name)
        if col_name != self.cur_col:
            self.cur_col = col_name
            self.col_values = prep.get_col_values(col_name)

        self.hist_plot.hist(self.col_values)
        self.hist_plot.grid()
        self.fig.canvas.draw()

    def reset(self):
        for ax in self.axes:
            ax.cla()
        self.fig.canvas.draw()

    # def analyze_column_data(self, idx):
    #     log.debug('>>>>> idx:%d, column:%s' % (idx, colname))
    #     col_map = np.unique(self.X[colname])
    #     if len(col_map) <= 10:
    #         log.debug('col_map:%s' % (col_map))
    #         self.analyze_small_range_data(idx, colname, col_map)
    #     else:
    #         log.debug('col_map length:%d' % len(col_map))
    #         self.analyze_big_range_data(idx, colname, col_map)
    #
    #     self.axs[idx].grid()
    #     self.fig.canvas.draw()
    #
    # def analyze_small_range_data(self, col_map):
    #     width = 0.35
    #     col_survived = self.X[colname][self.idx_survived]
    #     col_died = self.X[colname][self.idx_died]
    #     count_survived = {}
    #     count_died = {}
    #     for value in col_map:
    #         count_survived[value] = np.sum(col_survived == value)
    #         count_died[value] = np.sum(col_died == value)
    #
    #     N = len(col_map)
    #     ind = np.arange(N)
    #
    #     self.axs[idx].cla()
    #     self.axs[idx].bar(ind, count_survived.values(), width, color=survived_color, label='Survived')
    #     self.axs[idx].bar(ind + width, count_died.values(), width, color=died_color, label='Died')
    #
    #     self.axs[idx].set_xlabel(colname, fontsize=12)
    #     self.axs[idx].set_ylabel('Number of people', fontsize=12)
    #     self.axs[idx].legend(loc='upper right')
    #     log.debug('ind + width:%s, col_map:%s' % (ind + width, col_map))
    #     self.axs[idx].set_xticks(ind + width)
    #     self.axs[idx].set_xticklabels(col_map)
    #
    #
    # def analyze_big_range_data(self, col_map):
    #     bincount = 0
    #     if colname == 'Fare':
    #         bincount = 25
    #         width = 20
    #     elif colname == 'Age':
    #         bincount = 100
    #
    #     col_surv = self.X[colname][self.idx_survived]
    #     col_died = self.X[colname][self.idx_died]
    #
    #     minval, maxval = min(col_surv), max(col_surv)
    #     log.debug('min:%s, max:%s' % (minval, maxval) )
    #     bins = np.linspace(minval, maxval, bincount)
    #
    #     count_surv, _ = np.histogram(col_surv, bins)
    #     count_died, _ = np.histogram(col_died, bins)
    #
    #     self.axs[idx].cla()
    #     if colname == 'Fare':
    #         self.axs[idx].bar(bins[:-1], np.log10(count_surv), width=width,
    #                           color=survived_color, label='Survived')
    #         self.axs[idx].bar(bins[:-1], -np.log10(count_died), width=width,
    #                           color=died_color, label='Died')
    #     elif colname == 'Age':
    #         self.axs[idx].bar(bins[:-1], np.log10(count_surv), color=survived_color,
    #                           label='Survived')
    #         self.axs[idx].bar(bins[:-1], -np.log10(count_died), color=died_color,
    #                           label='Died')
    #     self.axs[idx].set_ylabel('Number of people')
    #     self.axs[idx].set_xlabel(colname)
    #     self.axs[idx].set_yticks(range(-3, 4), (10**abs(k) for k in range(-3, 4)))
    #     self.axs[idx].set_yticklabels((10**abs(k) for k in range(-3, 4)))