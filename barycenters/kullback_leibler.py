import ot
import numpy as np
from barycenters import project_simplex

def evaluate(a,b,M):
    _b = b.copy()
    _b[_b == 0] = 1
    div = a / _b
    div[div < 0] = 0
    return -np.sum(np.log(div)*a)

def fixed_support_barycenter(B, eta=1, numItermax=100, stopThr=1e-9, verbose=True):
    a = ot.unif(B.shape[1])
    a_prev = a.copy()

    for k in range(numItermax):
        G = []
        for i in range(B.shape[0]):
            _b = B[i].copy()
            _b[_b == 0] = 1
            div = a / _b
            div[div < 0] = 0
            G.append(np.log(div))
        g = sum(G) / len(G)

        a = project_simplex(a - eta * g)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))
        a_prev = a.copy()

    return a