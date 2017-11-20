# -*- coding: utf-8 -*-

import logging as log

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from common.utils import *
from common.utils_ui import *
from gui.preprocess_views.column_info.dist_chart import CentralChart
from preprocess_data import preprocess as prep
from common import strings as strs


log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

survived_color = '#6699ff'
died_color = '#ff6666'

class DetailView(QWidget):
    def __init__(self, parent=None):
        super(DetailView, self).__init__(parent)
        log.debug('start')

        # data
        self.ctrl_view = None
        self.cur_col = None
        self.col_values = None
        self.label_values = None
        self.label_index_map = None
        self.label_color = ['b', 'r', 'g', 'c', 'm', 'y']

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
        self.info_view = CentralChart(self.cur_col)
        layout.addWidget(self.info_view, 0, 1)

        log.debug('parent: %s', self.parent)

    def reset(self):
        for ax in self.axes:
            ax.cla()
        self.fig.canvas.draw()

    def load_col_values(self, col_name):
        if col_name != self.cur_col:
            self.cur_col = col_name
            self.col_values = prep.get_col_values(col_name)

    def load_label_values(self):
        if self.label_values is None:
            self.label_values = prep.get_label_values()

        unique_labels = np.unique(self.label_values)
        self.label_index_map = dict()
        for label in unique_labels:
            self.label_index_map.setdefault(label, 0)
            self.label_index_map[label] = (self.label_values == label)

    def show_detail_info(self, col_name, property_ctrl):
        log.debug('start - %s' % col_name)
        self.ctrl_view = property_ctrl
        self.load_col_values(col_name)

        # 컬럼 값 히스토그램
        if self.col_values is not None:
            self.show_hist_plot(col_name)
        else:
            log.debug(strs.error_no_column_values)

        # label(y값)과 연계된 챠트
        self.load_label_values()

        if self.label_values is not None:
            self.show_chart_plot(col_name)
        else:
            log.debug(strs.error_no_label_value)

        # Information view
        self.info_view.show_desc_info(self.cur_col)

    def show_hist_plot(self, col_name):
        # 현재 플롯 클리어
        self.load_col_values(col_name)
        self.hist_plot.cla()
        self.fig.canvas.draw()

        # 선택된 컬럼에 대한 플로팅
        try:
            self.hist_plot.hist(self.col_values.dropna())
            self.hist_plot.grid()
            self.fig.canvas.draw()
        except TypeError:
            log.debug(strs.error_type_error_exception)

    # 각 레이블별 컬럼값을 구분하여 보여주기
    def show_chart_plot(self, col_name):
        log.debug('>>>>>> start')
        log.debug('col name: %s', col_name)
        self.load_col_values(col_name)
        col_info = prep.get_col_info(col_name)
        if col_info[strs.col_data_range] == strs.col_data_range_small:
            col_map = np.unique(self.col_values)
            log.debug('col_map:{0}' % col_map)
            self.analyze_small_range_data(col_map)
        elif col_info[strs.col_data_range] == strs.col_data_range_big:
            col_map = np.unique(self.col_values)
            log.debug('col_map length:%d' % len(col_map))
            self.analyze_big_range_data(col_map)
        else:
            log.debug('col_data_range_none:{0}'.format(strs.col_data_range_none))
        # if len(col_map) <= 10:
        #     log.debug('col_map:%s' % col_map)
        #     self.analyze_small_range_data(col_map)
        # else:
        #     log.debug('col_map length:%d' % len(col_map))
        #     self.analyze_big_range_data(col_map)

        self.chart_plot.grid()
        self.fig.canvas.draw()

    def analyze_small_range_data(self, col_map):
        width = 0.35
        self.load_label_values()
        counts_of_each_label = {}

        for label_val in self.label_index_map:
            counts_of_each_label.setdefault(label_val, [])
            for value in col_map:
                counts_of_each_label[label_val].append(np.sum(self.col_values[self.label_index_map[label_val]] == value))

        N = len(col_map)
        ind = np.arange(N)

        self.chart_plot.cla()
        i = 0
        for label_val in self.label_index_map:
            log.debug('label: {0}'.format(label_val))
            self.chart_plot.bar(ind+width*i,
                                counts_of_each_label[label_val],
                                width,
                                color=self.label_color[i],
                                label=label_val)
            i += 1

        self.chart_plot.set_xlabel(self.cur_col, fontsize=12)
        self.chart_plot.set_ylabel('Number of people', fontsize=12)
        self.chart_plot.legend(loc='upper right')
        log.debug('ind + width:%s, col_map:%s' % (ind + width, col_map))
        self.chart_plot.set_xticks(ind + width)
        self.chart_plot.set_xticklabels(col_map)


    def analyze_big_range_data(self, col_map):
        self.load_label_values()

        try:
            bins = get_fd_bins(self.col_values)
            width = 1.0 / ((len(self.label_index_map) + 1) * len(bins))

            self.chart_plot.cla()
            index = 0
            for label_val in self.label_index_map:
                hist_vals, _ = np.histogram(self.col_values[self.label_index_map[label_val]], bins)
                self.chart_plot.bar(bins[:-1], hist_vals, width=width, color=get_color(index))
                index += 1
                width = width * 2

            self.chart_plot.set_xlabel(self.cur_col, fontsize=12)
            self.chart_plot.set_ylabel('Number of people', fontsize=12)
            self.chart_plot.legend(loc='upper right')
            self.chart_plot.set_xticks(bins + width)
            self.chart_plot.set_xticklabels(bins)
        except Exception as e:
            self.ctrl_view.exclude_column(self.cur_col)
            log.debug('exception occured')

    # 참고 소스
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
