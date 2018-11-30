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
    bad_flags = 1
    bmask[:, 0] = bad_flags
    bmask[:, -2] = 2
    symmetrize_bmask(bmask=bmask, bad_flags=bad_flags)

    assert np.array_equal(
        bmask,
        [[1, 0, 2, 0],
         [1, 0, 2, 0],
         [1, 0, 2, 0],
         [1, 1, 3, 1]])
