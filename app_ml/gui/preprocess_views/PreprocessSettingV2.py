import logging as log
import sys
from functools import partial

from PyQt5.QtWidgets import *

import preprocess_data.preprocess as prep
from common import strings as strs

log.basicConfig(format=strs.log_format,level=log.DEBUG,stream=sys.stderr)

class PreprocessSettingViewV2(QDockWidget):
    def __init__(self, main_view=None, parent=None):
        super(PreprocessSettingViewV2, self).__init__(parent)
        self.central_view = main_view
        self.initialized = False;

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.setWidget(widget)

        self.test_btn = QPushButton(strs.btn_test, parent=self)
        self.test_btn.clicked.connect(self.on_test_clicked)

    def on_test_clicked(self):
        log.debug('>>>>>> start')
