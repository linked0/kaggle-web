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
from gui.preprocess_views.ColumnDetailInfo import ColumnInfoView

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

survived_color = '#6699ff'
died_color = '#ff6666'

class ColumnDetailView(QWidget):
    def __init__(self, parent=None):
        super(ColumnDetailView, self).__init__(parent)
        log.debug('start')

        # data
        self.cur_col = None
        self.col_values = None
        self.label_values = None

        # layout
        layout = QGridLayout()
        self.setLayout(layout)

        # distribution plotting view
        self.fig, self.axes = plt.subplots(2, 1)
        self.hist_plot = self.axes[0]
        self.chart_plot = self.axes[1]
        self.plot_canvas = FigureCanvas(self.fig)
        layout.addWidget(self.plot_canvas, 0, 0)

        # detail information view (Missing Value Count, Type, Value Set, Desc)
        self.info_view = ColumnInfoView(self.cur_col)
        layout.addWidget(self.info_view, 0, 1)

        log.debug('parent: %s', self.parent)

    def reset(self):
        for ax in self.axes:
            ax.cla()
        self.fig.canvas.draw()

    def show_detail_info(self, col_name):
        log.debug('start - %s' % col_name)

        # 컬럼 값 히스토그램
        if col_name != self.cur_col:
            self.cur_col = col_name
            self.col_values = prep.get_col_values(col_name)
        if self.col_values is not None:
            self.show_hist_plot()
        else:
            log.debug(strs.error_no_column_values)

        # label(y값)과 연계된 챠트
        if self.label_values is None:
            self.label_values = prep.get_label_values()

        if self.label_values is not None:
            self.show_chart_plot()
        else:
            log.debug(strs.error_no_label_values)

        # Information view
        self.info_view.show_info(self.cur_col)

    def show_hist_plot(self):
        # 현재 플롯 클리어
        self.hist_plot.cla()
        self.fig.canvas.draw()

        # 선택된 컬럼에 대한 플로팅
        try:
            self.hist_plot.hist(self.col_values.dropna())
            self.hist_plot.grid()
            self.fig.canvas.draw()
        except TypeError:
            log.debug(strs.error_type_error_exception)

    def show_chart_plot(self):
        pass

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
