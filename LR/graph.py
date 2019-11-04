import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

frequency = 100000

#plotting the cost function

def init_cost(fn):
    plt.ion()
    fig = plt.figure(0, figsize=(8, 4))

    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.set_ylim(-1.0, 1.0)

    plt.xlabel('n Times')
    plt.ylabel('J(Θ)')
    plt.title(f'"{fn}" dataset\'s cost function')

    xdata, ydata = [], []
    line, = ax.plot(xdata, ydata, 'r-')

    return (fig, ax, xdata, ydata, line)

def replot_cost(fig, ax, line, nTimes, xdata, ydata, k, LL):

    _, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if k > xmax:
        xmax = nTimes if 2 * k > nTimes else 2 * k
    if LL > ymax:
        ymax = 1.2 * LL
    if LL < ymin:
        ymin = 1.2 * LL

    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)

    xdata.append(k + 1)
    ydata.append(LL)
    line.set_data(xdata, ydata)

    if k % (nTimes / frequency) == 0:
        fig.canvas.draw()
        fig.canvas.flush_events()

# Fitting the actual line to the data

def init_line(X, y, thetas):
    fig = plt.figure(1, figsize=(8,4))

    ax = fig.add_subplot(1, 1, 1)
    ax.grid()

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Data scatterplot and a fitted line')

    xdata = X[:, 1]
    ydata = np.matmul(X, thetas)

    line, = ax.plot(xdata, ydata, 'r-')
    data_plot = ax.scatter(xdata, y)

    return (fig, ax, xdata, line)

def replot_line(fig, ax, xdata, line, X, thetas, k, nTimes):

    ydata = np.matmul(X, thetas)
    line.set_data(xdata, ydata)

    if k % (nTimes / frequency) == 0:
        fig.canvas.draw()
        fig.canvas.flush_events()


