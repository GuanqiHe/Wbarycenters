import numpy as np

def evaluate(a,b):
    return np.linalg.norm(a-b)


def fixed_support_barycenter(B):

    return np.mean(B, axis=0)