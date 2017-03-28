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

log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)


def input_widget_horizontal(title_text, min_title_height=None, min_cont_height=None):
    widget = QWidget()
    layout = QHBoxLayout()
    widget.setLayout(layout)

    title = QLabel(title_text)
    if min_title_height is not None:
        title.setMinimumHeight(min_title_height)
    layout.addWidget(title)

    cont = QLabel()
    cont.setStyleSheet('border: 1px solid black;')
    if min_cont_height is not None:
        cont.setMinimumHeight(min_cont_height)
    layout.addWidget(cont)

    return widget


def input_widget_vertical(title_text, min_title_height=None, min_cont_height=None):
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)

    title = QLabel(title_text)
    title.setStyleSheet('border: 1px')
    if min_title_height is not None:
        title.setMinimumHeight(min_title_height)
    layout.addWidget(title)

    cont = QLabel()
    cont.setStyleSheet('border: 1px solid black;')
    if min_cont_height is not None:
        cont.setMinimumHeight(min_cont_height)
    layout.addWidget(cont)
    return widget