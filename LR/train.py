import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()

from LR.readFile import readFile
from LR.graph import init_cost, replot_cost, init_line, replot_line

def train(fn):
    nTimes = 10000000
    rate = 5e-25
    data, n, m = readFile(fn)

    X = data[:, :-1]
    y = data[:, -1]
    thetas = np.random.rand(m)
    gradient = np.zeros(m)

    fig_cost, ax_cost, xdata_cost, ydata_cost, curve = init_cost('Movie Revenue')
    fig_line, ax_line, xdata_line, line, = init_line(X, y,thetas)

    for k in range(nTimes):
        h = np.matmul(X, thetas)
        delta = y - h
        gradient = -2 * np.sum(X * delta[:, np.newaxis], axis=0)
        thetas -= rate * gradient

        J = np.sum(delta * delta)

        replot_cost(fig_cost, ax_cost, curve, nTimes, xdata_cost, ydata_cost, k, J)
        replot_line(fig_line, ax_line, xdata_line, line, X, thetas, k, nTimes)

    plt.ioff()
    plt.savefig('LR/line_graph.png')
    plt.savefig('LR/final_line.png')

    return thetas