import numpy as np
import pytest

from .._slice_data import (
    symmetrize_bmask,
    symmetrize_weight,
    build_slice_locations)


def test_build_slice_locations():
    row, col, start_row, start_col = build_slice_locations(
        central_size=20, buffer_size=10, image_width=100)

    for i in range(4):
        for j in range(4):
            ind = j + 4*i
            assert row[ind] == i*20 + 10 + 9.5
            assert col[ind] == j*20 + 10 + 9.5
            assert start_row[ind] == i * 20
            assert start_col[ind] == j * 20


def test_build_slice_locations_error():
    with pytest.raises(ValueError):
        build_slice_locations(
            central_size=15, buffer_size=10, image_width=100)


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
