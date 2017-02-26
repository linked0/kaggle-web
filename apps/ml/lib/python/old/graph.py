__author__ = 'linked0'
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pandas import DataFrame
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# def plot(li):
#     li.insert(0,0)
#     li.append(0)
#     ax.plot(li, '.')
#     fig.canvas.draw()

def new_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    return fig, ax

def new_plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    return fig, ax

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

def sample_poly3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

    xs = np.arange(0, 10, 0.4)
    verts = []
    zs = [0.0, 1.0, 2.0, 3.0]
    for z in zs:
        ys = np.random.rand(len(xs))
        ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))

    poly = PolyCollection(verts, facecolors = [cc('r'), cc('g'), cc('b'),
                                               cc('y')])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 4)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 1)

    plt.show()

def sample_wire3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()

