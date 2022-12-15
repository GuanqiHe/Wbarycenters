import ot
import numpy as np

from barycenters import project_simplex

def evaluate(a,b,M):
    cost, ret = ot.emd(a, b, M, log=True)
    return ret['cost']


def fixed_support_barycenter(B, M, weights=None, eta=10, numItermax=100, stopThr=1e-9, verbose=False, norm='max'):
    a = ot.unif(B.shape[1])
    a_prev = a.copy()
    weights = ot.unif(B.shape[0]) if weights is None else weights
    if norm == "max":
        _M = M / np.max(M)
    elif norm == "median":
        _M = M / np.median(M)
    else:
        _M = M

    for k in range(numItermax):
        potentials = []
        for i in range(B.shape[0]):
            _, ret = ot.emd(a, B[i], _M, log=True)
            potentials.append(ret['u'])
        
        # Calculates the gradient
        f_star = sum(potentials) / len(potentials)

        # Mirror Descent
        a = a * np.exp(- eta * f_star)

        # Projection
        a = project_simplex(a)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))

        # Update previous a
        a_prev = a.copy()
    return a