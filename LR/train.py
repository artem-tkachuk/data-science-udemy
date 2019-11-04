import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()

from LR.readFile import readFile
from LR.graph import init_cost, replot_cost, init_line, replot_line

def train(fn):
    # number of training iterations
    nTimes = 10000000
    # step size (learning rate)
    rate = 5e-25
    # data, # of training examples, # of input features + 1
    data, n, m = readFile(fn)

    # slice input features only
    X = data[:, :-1]
    # slice targets only
    y = data[:, -1]
    # initial values for parameters
    thetas = np.random.rand(m)
    # initialize gradients
    gradient = np.zeros(m)

    # initialize the objects for cost function continuous replotting
    fig_cost, ax_cost, xdata_cost, ydata_cost, curve = init_cost('Movie Revenue')
    # initialize the objects for the continuous replotting of the fitted line
    fig_line, ax_line, xdata_line, line, = init_line(X, y, thetas)

    # the actual training process
    for k in range(nTimes):
        h = np.matmul(X, thetas)
        delta = y - h
        # get the partial derivatives
        gradient = -2 * np.sum(X * delta[:, np.newaxis], axis=0)
        # update the parameters
        thetas -= rate * gradient

        # cost function's current value
        J = np.sum(delta * delta)

        # replot the cost function and the fitted line
        replot_cost(fig_cost, ax_cost, curve, nTimes, xdata_cost, ydata_cost, k, J)
        replot_line(fig_line, ax_line, xdata_line, line, X, thetas, k, nTimes)

    # this line is needed for graphs to stay afterwards
    plt.ioff()
    # save the .png of the graphs for later reference
    plt.savefig('LR/line_graph.png')
    plt.savefig('LR/final_line.png')

    return thetas