import sys
import os
import logging as log
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import common.strings as strs

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

class RLSubView(QDockWidget):
    def __init__(self, parent=None):
        super(RLSubView, self).__init__(parent)

        self.titlelabel = QLabel('Reinforcement Learning SubView')
        self.setWidget(self.titlelabel)