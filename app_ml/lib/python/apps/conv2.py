#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import math
from os import listdir
from os.path import isfile, join

def read_data(filename):
    xmlTag = '<?xml'

    inFile = open(filename)
    head = inFile.read(10)
    if head.find(xmlTag) == -1:
        inFile.close()
        sensor_data = pd.read_csv(filename)
        return sensor_data

    startTag = '<string>'
    endTag = '</string>'
    outFileName = filename+'.out'
    outFile = open(outFileName, 'w')

    for line in inFile:
        startIndex = line.find(startTag)
        if startIndex != -1:
            endIndex = line.find(endTag)
            outFile.write(line[startIndex+len(startTag) : len(line) - len(endTag) - 1])
            outFile.write('\n')

    inFile.close()
    outFile.close()
    shutil.move(outFileName, filename)
    sensor_data = pd.read_csv(filename)
    return sensor_data

def _show_plot(dt, filename):
    dt2 = dt.ix[dt['mag_val'].notnull()][['mag_val']]
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    plt.plot(dt2, '.')

    ax1.text(1, dt.mag_val.max()+1, filename, family='monospace', color='red', fontsize=15)

    dt_desc = dt.ix[dt.desc.notnull()]
    for idx in dt_desc.index:
        ax1.annotate(dt.desc[idx], xy=(idx, dt.mag_val[idx]+1), xytext=(idx, dt.mag_val[idx]+4),
                     arrowprops=dict(facecolor='black'), horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()
    return fig, ax1, dt2

def show_plot2(filename, order):
    global dt
    dt = read_data(filename)
    print type(dt)
    fig, ax1, dtmag = _show_plot(dt, filename)

    ax2, dtx = _show_plot_x(fig, dt, filename)
    show_interpolation(fig, ax2, dtx, order)

    ax3, dty = _show_plot_y(fig, dt, filename)
    show_interpolation(fig, ax3, dty, order)

    ax4, dtz = _show_plot_z(fig, dt, filename)
    show_interpolation(fig, ax4, dtz, order)

    # return dt
    # return section_sum()
    #return fig, ax1, dt

def _show_plot_x(fig, dt, filename):
    dt2 = dt.ix[dt['mag_x'].notnull()][['mag_x']]
    ax1 = fig.add_subplot(2,2,2)
    plt.plot(dt2, '.')

    ax1.text(1, dt.mag_x.max()+1, filename+": mag_x", family='monospace', color='red', fontsize=14)

    dt_desc = dt.ix[dt.desc.notnull()]
    for idx in dt_desc.index:
        full_desc = dt.desc[idx]+":"+str(round(dt.mag_x[idx], 1))
        ax1.annotate(full_desc, xy=(idx, dt.mag_x[idx]+1), xytext=(idx, dt.mag_x[idx]+4),
                     arrowprops=dict(facecolor='black'), horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()
    return ax1

def _show_plot_y(fig, dt, filename):
    dt2 = dt.ix[dt['mag_y'].notnull()][['mag_y']]
    ax1 = fig.add_subplot(2,2,3)
    plt.plot(dt2, '.')

    ax1.text(1, dt.mag_y.max()+1, filename+": mag_y", family='monospace', color='red', fontsize=14)

    dt_desc = dt.ix[dt.desc.notnull()]
    for idx in dt_desc.index:
        full_desc = dt.desc[idx]+":"+str(round(dt.mag_y[idx], 1))
        ax1.annotate(full_desc, xy=(idx, dt.mag_y[idx]+1), xytext=(idx, dt.mag_y[idx]+4),
                     arrowprops=dict(facecolor='black'), horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()
    return ax1

def _show_plot_z(fig, dt, filename):
    dt2 = dt.ix[dt['mag_z'].notnull()][['mag_z']]
    ax1 = fig.add_subplot(2,2,4)
    plt.plot(dt2, '.')

    ax1.text(1, dt.mag_z.max()+1, filename+": mag_z", family='monospace', color='red', fontsize=14)

    dt_desc = dt.ix[dt.desc.notnull()]
    for idx in dt_desc.index:
        full_desc = dt.desc[idx]+":"+str(round(dt.mag_z[idx], 1))
        ax1.annotate(full_desc, xy=(idx, dt.mag_z[idx]+1), xytext=(idx, dt.mag_z[idx]+4),
                     arrowprops=dict(facecolor='black'), horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()
    return ax1

def plot(filename):
    dt = read_data(filename)
    print type(dt)
    fig, ax1, dtmag = _show_plot(dt, filename)
    ax2 = _show_plot_x(fig, dt, filename)
    ax3 = _show_plot_y(fig, dt, filename)
    ax4 = _show_plot_z(fig, dt, filename)

    print "return show_plot"
    return fig, [ax1,ax2,ax3,ax4], dt

def plot_z(filename):
    dt = read_data(filename)
    dt2 = dt.ix[dt['mag_z'].notnull()][['mag_z']]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.plot(dt2, '.')

    ax1.text(1, dt.mag_z.max()+1, filename+": mag_z", family='monospace', color='red', fontsize=14)

    dt_desc = dt.ix[dt.desc.notnull()]
    for idx in dt_desc.index:
        full_desc = dt.desc[idx]+":"+str(round(dt.mag_z[idx], 1))
        ax1.annotate(full_desc, xy=(idx, dt.mag_z[idx]+1), xytext=(idx, dt.mag_z[idx]+4),
                     arrowprops=dict(facecolor='black'), horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()

def plot_z2(filename, order):
    dt = read_data(filename)
    dt2 = dt.ix[dt['mag_z'].notnull()][['mag_z']]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.plot(dt2, '.')

    ax1.text(1, dt.mag_z.max()+1, filename+": mag_z", family='monospace', color='red', fontsize=14)

    dt_desc = dt.ix[dt.desc.notnull()]
    for idx in dt_desc.index:
        full_desc = dt.desc[idx]+":"+str(round(dt.mag_z[idx], 1))
        ax1.annotate(full_desc, xy=(idx, dt.mag_z[idx]+1), xytext=(idx, dt.mag_z[idx]+4),
                     arrowprops=dict(facecolor='black'), horizontalalignment='left', verticalalignment='top')

    show_interpolation(fig, ax1, dt2, order)

from decimal import *

def plot_z3(filename, fig, pos, desc):
    dt = read_data(filename)
    dt2 = dt.ix[dt['mag_z'].notnull()][['mag_z']]

    ax1 = fig.add_subplot(2,2,pos)
    plt.plot(dt2, '.')

    diff = dt2.max() - dt2.min()
    diff = round(diff, 2)
    diff = math.fabs(diff)
    ax1.set_title(desc+", Max-Min:" + str(diff))
    ax1.get_xaxis().set_visible(False)
    fig.canvas.draw()

    interline(fig, ax1, dt2)

def interline(fig, ax1, dt2):
    starendnum = 10
    generalnum = 20
    nodepoints = 20
    

import re
import shutil

def plot_zs(foldername):
    datafiles = [f for f in listdir(foldername) if isfile(join(foldername, f))]
    for f in datafiles:
        if re.match(".+_r[0-9]_.+", f) != None:
            dst = re.sub(r"(.+)_r([0-9])_(.+)", r"\1_r0\2_\3", f)
            shutil.move(join(foldername, f), join(foldername, dst))

    datafiles = [f for f in listdir(foldername) if isfile(join(foldername, f))]
    i = 1
    fig  = plt.figure()
    for f in datafiles:
        li = re.split("_", f)
        plot_z3(join(foldername, f), fig, i, li[4]+"_"+li[5])
        i += 1

# def plot_zs(foldername):
#     datafiles = [f for f in listdir(foldername) if isfile(join(foldername, f))]
#     for f in datafiles:
#         if re.match(".+_r[0-9]_.+", f) != None:
#             dst = re.sub(r"(.+)_r([0-9])_(.+)", r"\1_r0\2_\3", f)
#             shutil.move(join(foldername, f), join(foldername, dst))
#
#     datafiles = [f for f in listdir(foldername) if isfile(join(foldername, f))]
#     i = 1
#     fig  = plt.figure()
#     for f in datafiles:
#         print f
#         li = re.split("_", f)
#         a = i / 5
#         if a > 0 and i % 5 == 0:
#             a = a - 1
#
#         b = i % 5
#         if b == 0:
#             b = 5
#
#         if i <= 25:
#             print "plot pos: " + str(b) + ", " + str(a) + ", i:" + str(i)
#             plot_z3(join(foldername, f), fig, (b-1)*5 + (a+1), li[4]+"_"+li[5])
#         i += 1


def intToStr(n, base, alphabet):
    def toStr(n, base, alphabet):
        return alphabet[n] if n < base else toStr(n//base,base,alphabet) + alphabet[n%base]
    return ('-' if n < 0 else '') + toStr(abs(n), base, alphabet)

def show_interpolation(fig, ax, dta, order):
    print "Max Value: ", dta.max()
    # print dta.shape

    y = dta
    rows = y.shape[0]
    x = np.arange(0, rows)
    fxp = np.polyfit(x, y, order)
    coeffs = [fxp[i][0] for i in range(0, order+1)]
    # print fxp
    #print coeffs
    fx = np.poly1d(coeffs)
    ax.plot(x, fx(x), linewidth = 4, color="red")
    fig.canvas.draw()

############################################################
# 방향별 보정을 위한 분석 스크립트
############################################################
def _show_orient(fig, dt, filename, colname, plotidx):
    piDesc = ['0', 'pi/2', 'pi' , 'pi3/2', '2pi']
    dt2 = dt.ix[dt[colname].notnull()][[colname, 'orient']]
    if colname == 'mag_val':
        fig = plt.figure()
    ax1 = fig.add_subplot(2,2,plotidx)
    plt.plot(dt2[colname], '.')

    diffMinMax = dt2[colname].max() - dt2[colname].min()
    posByDiff = diffMinMax/30
    posByDiff2 = diffMinMax/5
    findIdx = 0;
    diff = 0.1
    incPi = math.pi/2;
    # print dt2
    # for idx in dt2.index:
    #     # print piDesc[findIdx]
    #     findPi = incPi * findIdx
    #     if dt2.orient[idx] >= findPi-diff and dt2.orient[idx] <= findPi+diff:
    #         ax1.annotate(piDesc[findIdx] + ": " + str(dt2[colname][idx]) , xy=(idx, dt2[colname][idx]+posByDiff),
    #                      xytext=(idx, dt2[colname][idx]+posByDiff2),
    #                      arrowprops=dict(facecolor='red'), horizontalalignment='left',
    #                                      verticalalignment='top')
    #         findIdx += 1
    #         if findIdx == 4:
    #             break

    fig.canvas.draw()
    return fig, ax1, dt2

def ori(filename):
    dt = read_data(filename)
    dt2 = DataFrame(columns=dt.columns)
    foundStart = False
    foundEnd = False
    dt2Idx = 0;
    for rowIdx in range(dt.shape[0]):
        dtRow = dt.loc[rowIdx]
        if foundStart == False and dtRow.orient < math.pi/2:
            foundStart = True
            print "foundStart - idx: %d" % rowIdx
        if foundEnd == False and foundStart == True and rowIdx > dt.shape[0]/2 and dtRow.orient < math.pi/2:
            foundEnd = True
            print "foundEnd - idx: %d" % rowIdx

        if foundStart == True and foundEnd == False:
            dt2.loc[dt2Idx] = dtRow
            dt2Idx += 1

    dt_desc = dt[dt.desc.notnull()].desc[0]

    fig, ax1, dtmag = _show_orient(None, dt2, filename, 'mag_val', 1)
    fig, ax2, dtx = _show_orient(fig, dt2, filename, 'mag_x', 2)
    fig, ax3, dty = _show_orient(fig, dt2, filename, 'mag_y', 3)
    fig, ax4, dtz = _show_orient(fig, dt2, filename, 'mag_z', 4)

    fig.text(0.1, 0.95, filename+"-"+dt_desc, family='monospace', color='black', fontsize=15)
    fig.canvas.draw()

    # return fig, ax1, dt, dt2

############################################################
# 기타 참고 코드
############################################################
def show_wire3d(dt):
    dtMagVal = dt.mag_val

    fig, ax = new_plot_3d()

def show_plot3d(dt1, dt2, dt3):
    fig, ax = new_plot_3d()

    xs = [i for i in range(len(dt1))]
    ys1 = [0 for i in range(len(dt1))]
    ys2 = [132 for i in range(len(dt1))]
    ys3 = [243 for i in range(len(dt1))]

    ax.scatter(xs, ys1, list(dt1.mag_val), color='red')
    ax.scatter(xs, ys2, list(dt2.mag_val), color='green')
    ax.scatter(xs, ys3, list(dt2.mag_val), color='blue')

def show_plotwire(dt1, dt2, dt3):
    fig, ax = new_plot_3d()

    X = [None] * len(dt1)
    Y = [None] * len(dt1)
    Z = [[None] * len(dt1)] * len(dt1)

    for i in range(len(dt1)):
        X[i] = [j for j in range(len(dt1))]
        Y[i] = [i for k in range(len(dt1))]

    Z[0] = list(dt1.mag_val)
    Z[132] = list(dt2.mag_val)
    Z[243] = list(dt3.mag_val)

    # First Range
    for i in range(1, 132):
        for j in range(len(dt1)):
            div = (Z[132][j] - Z[0][j])/132
            print "first div ", div
            Z[i][j] = Z[0][j] + (div * i)

    # Second Range
    for i in range(133, 244):
        for j in range(len(dt1)):
            div = (Z[243][j] - Z[132][j])/110
            Z[i][j] = Z[132][j] + (div * (i-133))

    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()

    return X, Y, Z

def new_plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    return fig, ax

# Sample 3D plotting
def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

def sample_scatter3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zl, zh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def sample_wire3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()

    return X, Y, Z

def get_dt():
    global dt
    return dt

def section_sum():
    global dt
    sumli = []
    idx = dt.ix[dt.desc.notnull()].index
    for i in range(len(idx)-1):
        sumli.append(dt[idx[i]:idx[i+1]-1].mag_val.sum())

    return sumli

def keep_code():
    i = 0
    # ax = fig.add_subplot(111, projection='3d')
    # filename = sys.argv[1]

def get_peak(dt):
    peaks = []
    findUpPeak = True
    prev = 0
    peakCand = 0
    upCnt = 0
    downCnt = 0
    cntDelim = 30
    dtVal = dt['mag_val']
    print len(dtVal)
    for i in range(len(dtVal)):
        # print dtVal[i]
        if findUpPeak == True:
            if dtVal[i] >= prev:
                peakCand = i
                prev = dtVal[i]
                downCnt = 0
            else:
                downCnt += 1

            if downCnt > cntDelim:
                peaks.append(peakCand)
                print "up peak: ", peakCand, ", val: ", dtVal[peakCand]
                findUpPeak = False
                upCnt = 0

        else:
            if dtVal[i] <= prev:
                peakCand = i
                prev = dtVal[i]
                upCnt = 0
            else:
                upCnt += 1

            if upCnt > cntDelim:
                peaks.append(peakCand)
                print "down peak: ", peakCand, ", val: ", dtVal[peakCand]
                findUpPeak = True
                downCnt = 0

    return peaks

