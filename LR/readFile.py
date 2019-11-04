import io
import numpy as np

def readFile(fn):
    f = open(fn, 'r')

    m = int(f.readline()) + 1
    n = int(f.readline())

    lines = f.read().replace(",", " ")
    data = np.genfromtxt(io.StringIO(lines))
    data = np.insert(data, 0, 1, axis=1)

    return (data, n, m)