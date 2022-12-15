import numpy as np
from skimage.transform import resize


def project_simplex(x):
    """Project Simplex

    Projects an arbitrary vector :math:`\mathbf{x}` into the probability simplex, such that,

    .. math:: \tilde{\mathbf{x}}_{i} = \dfrac{\mathbf{x}_{i}}{\sum_{j=1}^{n}\mathbf{x}_{j}}

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Numpy array of shape (n,)

    Returns
    -------
    y : :class:`numpy.ndarray`
        numpy array lying on the probability simplex of shape (n,)
    """
    x[x < 0] = 0
    if np.isclose(sum(x), 0):
        y = np.zeros_like(x)
    else:
        y = x.copy() / sum(x)
    return y


def create_digits_image(images, labels, digit=0, n_digits=15, is_distribution=True):
    batch = images[np.where(labels==digit)[0]]
    rows, cols = images[0].shape
    images = []
    for i in range(n_digits):
        img = batch[i]
        if is_distribution:
            img = img / np.sum(img)
        images.append(img.reshape(1, rows, cols))
    images = np.array(images).reshape(-1, rows * cols)
    return images