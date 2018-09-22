"""
File: nyu.py
Author: Nrupatunga
Email: nrupatunga@whodat.in
Github: https://github.com/nrupatunga
Description: python class for processing nyu dataset
"""

import numpy as np
from helper import indices as find
from mean_shift import (mean_shift, meanshift_dist2)


def get_surface_3d(norms3d, ntrails=100):
    """find surfaces
    """

    h = 0.01
    epsilon = 0.001
    thresh = 1 + 50 / 77.
    mode, score = mean_shift(norms3d, ntrails, h, epsilon, meanshift_dist2)
    idx = find(score, lambda x: x > thresh)
    mode = mode[:, idx]
    xyz_surf = np.zeros(mode.shape)
    for i in range(mode.shape[1]):
        rep_norm = abs(np.matmul(mode[:, i], norms3d))
        xyz_surf[:, i] = norms3d[:, np.argmax(rep_norm)]

    return xyz_surf


def get_room_directions(norms3d=None):
    """get room directions which are orthogonal to each other

    Args:
        norms3d: normal map
    """

    xyz_surf = get_surface_3d(norms3d.T)

    xyz_cand = xyz_surf
    idx = find(abs(xyz_cand[1, :]), lambda x: x > 0.8)
    ty = xyz_cand[:, idx]

    Nx = np.zeros((ty.shape[1] * 10, 3))
    Ny = np.zeros((ty.shape[1] * 10, 3))
    Nz = np.zeros((ty.shape[1] * 10, 3))

    maxpery = 10
    count = 0
    scores = np.zeros((ty.shape[1] * maxpery, 1))
    for i in range(ty.shape[1]):
        dist = abs(np.matmul(ty[:, i].T, xyz_cand))
        idx = find(dist, lambda x: x < 0.01)
        if len(idx) > maxpery:
            rp = np.random.permutation(len(idx))
            idx = [idx[ii] for ii in rp[0:maxpery].tolist()]

        for j in range(len(idx)):
            Ny[count, :] = ty[:, i]
            v2 = np.cross(xyz_cand[:, idx[j]], Ny[count, :])
            v1 = np.cross(Ny[count, :], v2)

            if np.square(v1[0]) > np.square(v2[0]):
                Nx[count, :] = v1
                Nz[count, :] = v2
            else:
                Nx[count, :] = v2
                Nz[count, :] = v1

            distX = np.matmul(norms3d, Nx[count, :].T)
            distY = np.matmul(norms3d, Ny[count, :].T)
            distZ = np.matmul(norms3d, Nz[count, :].T)

            scores[count] = np.sum(np.exp(-np.square(distX) / (0.01 * 0.01))) \
                + np.sum(np.exp(-np.square(distY) / (0.01 * 0.01))) \
                + np.sum(np.exp(-np.square(distZ) / (0.01 * 0.01)))

            count = count + 1

    scores = scores / norms3d.shape[0]

    # Make it so floor points up
    idx_ny = find(Ny[:, 1], lambda x: x > 0)

    if idx_ny is not None:
        Nx[idx_ny, :] = -Nx[idx_ny, :]
        Ny[idx_ny, :] = -Ny[idx_ny, :]
        Nz[idx_ny, :] = -Nz[idx_ny, :]

    idx = np.argmax(scores)
    R = Nx[idx, :], Ny[idx, :], Nz[idx, :]
    return R
