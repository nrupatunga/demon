"""
File: mean_shift.py
Author: Nrupatunga
Email: nrupatunga.tunga@gmail.com
Github: https://github.com/nrupatunga
Description: mean shift algorithms
"""

import numpy as np
from helper import indices as find

def meanshift_dist2(x1, x2):
    """distance metrics
    Args:
        x1: vector x1
        x2: vector x2

    Returns:
        dist: Euclidean distance
    """
    dist = np.sum((x1 - x2) * (x1 - x2), axis=0)
    return dist


def __m_hG(x, data, h, dist2):
    """compute mean shift of the data

    Args:
        x: initial x
        data: data dxN
        h: Gaussian window
        dist2: distance metrics

    Returns:
        ms: mean shift
        score: mean shift score
    """
    x_arr = x[np.newaxis].T
    x_arr = np.repeat(x_arr, data.shape[1], axis=1)

    num = dist2(x_arr, data)
    den = h * h

    e = np.divide(num, den)
    e = np.exp(-0.5 * e)

    score = np.sum(e)

    ms = np.matmul(data, e)
    ms = ms / score

    return ms, score


def __mean_shift_iteration(x, data, h, eps2, dist2):
    """ Iteratively find mean shift of the data
    Args:
        x: initial x
        data: data dxN
        h: Gaussian window
        eps2: threshold
        dist2: distance metrics

    Returns:
        mode: mean shift
        score: mean shift score
    """

    ms, score = __m_hG(x, data, h, dist2)
    while dist2(ms, x) >= eps2:
        x = ms
        ms, score = __m_hG(x, data, h, dist2)

    mode = ms
    return mode, score


def mean_shift(data, num_trails, h, epsilon, dist2):
    """find modes in the data
    Args:
        data: data dxN
        num_trails: number of trails
        h: Gaussian window
        epsilon: threshold
        dist2: distance metrics

    Returns:
        mode: modes
        score: scores of each mode

    """

    # PRINT.info('num_trails: {}, h: {}, epsilon: {}, dist2: {}'.format(
    #           num_trails, h, epsilon, dist2))

    d, N = data.shape
    num_trails = min(num_trails, N)
    rsp = np.random.permutation(N)
    rsp = rsp[0:num_trails]
    eps2 = epsilon * epsilon

    mode = np.zeros((d, num_trails))
    score = np.zeros((1, num_trails))

    for i in range(num_trails):
        x = data[:, rsp[i] - 1]
        mode[:, i], score[:, i] = __mean_shift_iteration(x, data, h, eps2, dist2)

    idxs = np.argsort(score)
    score = score[0][idxs[0]]
    mode = mode[:, idxs[0]]

    keep = np.ones((num_trails), dtype=bool)
    for i in range(0, num_trails - 1):
        x1 = np.repeat(mode[:, i][np.newaxis].T, num_trails - i - 1, axis=1)
        x2 = mode[:, i + 1:]
        d = dist2(x1, x2)
        idx = find(d, lambda x: x < (h * h))
        if len(idx):
            keep[i] = False

    score = score[keep]
    mode = mode[:, keep]

    return mode, score
