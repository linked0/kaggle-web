from __future__ import print_function
from __future__ import unicode_literals

import logging as log
import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from gui.preprocess_views.central_main import PreprocessCentralView
from gui.preprocess_views.setting_main import PreprocessSettingViewV2
from gui.TestMain import TestCentral
from gui.TestSub import TestSettingView
from gui.TrainSub import TrainSettingView
from gui.WebMain import WebBrowserCentral
from reinforcement_learning.rl_main import RLMainView
from reinforcement_learning.rl_sub import RLSubView

import common.control_view
import common.strings as strs
from gui.TrainMain import CentralWidgetTrainProcess
from examples.mlb import MlbEx

from preprocess_data import preprocess as prep

log.basicConfig(format=strs.log_format,level=log.DEBUG,stream=sys.stderr)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        layout = QVBoxLayout()
        centralwidget = QWidget()
        centralwidget.setLayout(layout)

        self.setWindowTitle('Training Machine')
        self.tabwidget = QTabWidget()

        # set data store
        # prep.set_data_name(prep.data_house_prices)

        # set for training model
        self.trainProcessView = CentralWidgetTrainProcess()
        self.train_selectionView = TrainSettingView()
        self.tabwidget.addTab(self.trainProcessView, strs.central_train_title)

        # set for preprocessing data
        self.preprocessView = PreprocessCentralView(main_view=self)
        self.preprocess_settingview = self.preprocessView.get_settingview()
        self.tabwidget.addTab(self.preprocessView, strs.central_preprocess_title)

        # set for reinforcement learning
        self.rl_mainview = RLMainView()
        self.rl_subview = RLSubView()
        self.tabwidget.addTab(self.rl_mainview, strs.central_lr_title)

        # set for web browser
        self.webbrowser = WebBrowserCentral()
        self.tabwidget.addTab(self.webbrowser, strs.central_web_title)

        # set for examples
        self.testCentral = TestCentral()
        self.test_settingview = TestSettingView()
        self.tabwidget.addTab(self.testCentral, strs.central_test_title)

        # set for MLB example
        # self.mlb_ex = MlbEx()
        # self.tabwidget.addTab(self.self.mlb_ex, strs.central_mlb_title)


        self.tabwidget.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tabwidget)

        # 디폴트 서브 뷰 세팅
        self.addDockWidget(Qt.RightDockWidgetArea, self.train_selectionView)
        self.current_rightdock = self.train_selectionView

        self.setCentralWidget(centralwidget)
        self.show()

        # status bar
        self.size_label = QLabel()
        self.size_label.setFrameStyle(QFrame.StyledPanel|QFrame.Sunken)
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.addPermanentWidget(self.size_label)
        status.showMessage('Good! You can change contents. Hyunjae! :)', 1000000)

        common.control_view.main_view = self
        common.control_view.central_tab_widget = self.tabwidget
        common.control_view.config_view = self.train_selectionView

    def on_tab_changed(self, p_int):
        log.debug('>>>>> %d' % p_int)
        self.removeDockWidget(self.current_rightdock)
        if p_int == 0:
            self.restoreDockWidget(self.train_selectionView)
            self.current_rightdock = self.train_selectionView
        elif p_int == 1:
            self.restoreDockWidget(self.preprocess_settingview)
            self.current_rightdock = self.preprocess_settingview
        elif p_int == 2:
            self.restoreDockWidget(self.rl_subview)
            self.current_rightdock = self.rl_subview
        elif p_int == 3:
            pass
        else:
            self.restoreDockWidget(self.test_settingview)
            self.current_rightdock = self.test_settingview

    def getTrainSubplot(self):
        return self.trainProcess.getTrainSubplot()
