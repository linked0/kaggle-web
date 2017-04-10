# -*- coding: utf-8 -*-
import sys
import os
import logging as log

from PyQt5.QtWidgets import *
from matplotlib.colors import ListedColormap
import common.strings as strs
import matplotlib.pyplot as plt
import numpy as np


log.basicConfig(format=strs.log_format, level=log.DEBUG, stream=sys.stderr)

def input_widget_horizontal(title_text, min_title_height=None, min_cont_height=None):
    widget = QWidget()
    layout = QHBoxLayout()
    widget.setLayout(layout)

    title = QLabel(title_text)
    if min_title_height is not None:
        title.setMinimumHeight(min_title_height)
    layout.addWidget(title)

    cont = QLineEdit()
    cont.setReadOnly(True)
    # cont.setStyleSheet('border: 1px solid black;')
    if min_cont_height is not None:
        cont.setMinimumHeight(min_cont_height)
    layout.addWidget(cont)

    return widget, cont


def input_widget_vertical(title_text, min_title_height=None, min_cont_height=None, max_cont_width=None):
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)

    title = QLabel(title_text)
    if min_title_height is not None:
        title.setMinimumHeight(min_title_height)
    layout.addWidget(title)

    cont = QTextEdit()
    cont.setReadOnly(True)
    # cont.setStyleSheet('border: 1px solid black;')
    if min_cont_height is not None:
        cont.setMinimumHeight(min_cont_height)
    if max_cont_width is not None:
        cont.setMaximumWidth(max_cont_width)
    layout.addWidget(cont)
    return widget, cont

markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')


def get_color_marker(index):
    return colors[index], markers[index]


def get_marker(index):
    return markers[index]


def get_color(index):
    return colors[index]