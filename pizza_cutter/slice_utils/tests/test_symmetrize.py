import numpy as np

from ..symmetrize import (
    symmetrize_bmask,
    symmetrize_weight)


def test_symmetrize_weight():
    weight = np.ones((5, 5))
    weight[:, 0] = 0
    symmetrize_weight(weight=weight)

    assert np.all(weight[:, 0] == 0)
    assert np.all(weight[-1, :] == 0)


def test_symmetrize_bmask():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    symmetrize_bmask(bmask=bmask)

    assert np.array_equal(
        bmask,
        [[1, 0, 2, 0],
         [3, 2, 2, 2],
         [1, 0, 2, 0],
         [1, 1, 3, 1]])


def test_symmetrize_bmask_angle():
    bmask = np.zeros((64, 64), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    bmask_orig = bmask.copy()
    bmasks = []
    for angle in [45, 90, 135, 180, 225, 270, 315]:
        bmask_r = bmask_orig.copy()
        symmetrize_bmask(bmask=bmask_r, angle=angle)
        bmasks.append(bmask_r)
        bmask |= bmask_r

    assert np.array_equal(bmask, np.rot90(bmask))
    assert np.array_equal(bmask, np.rot90(np.rot90(bmask)))
    assert np.array_equal(bmask, np.rot90(np.rot90(np.rot90(bmask))))

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=3, ncols=3)
        axs = axs.ravel()
        bmasks.append(bmask_orig)
        bmasks.append(bmask)
        for i, bm in enumerate(bmasks):
            ax = axs[i]
            ax.pcolormesh(bm)
            ax.set_aspect(1)
        import pdb
        pdb.set_trace()
