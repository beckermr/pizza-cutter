import numpy as np

import pytest

from ..flag import (
    slice_full_edge_masked,
    slice_has_flags,
    compute_masked_fraction)


@pytest.mark.parametrize(
    'inds',
    [(range(10), 0),  # left
     (range(10), -1),  # right
     (0, range(10)),  # top
     (-1, range(10))  # bottom
     ])
def test_slice_full_edge_masked(inds):
    """Make sure images with fully masked edges are all properly flagged."""
    weight = np.ones((10, 10))
    bmask = np.zeros((10, 10), dtype=np.int32)
    bmask[:, 0] = 4  # try a different flag
    bad_flags = np.int32(2**0)

    wgt = weight.copy()
    wgt[inds] = 0
    assert slice_full_edge_masked(
        weight=wgt, bmask=bmask, bad_flags=np.int32(0))

    bm = bmask.copy()
    bm[inds] = 2**0
    assert slice_full_edge_masked(
        weight=weight, bmask=bm, bad_flags=bad_flags)


@pytest.mark.parametrize(
    'inds',
    [(5, 0),  # left
     (5, -1),  # right
     (0, 5),  # top
     (-1, 5)  # bottom
     ])
def test_slice_full_edge_masked_one_pix(inds):
    """Make sure if not a fully masked edge, then image is not flagged."""
    weight = np.ones((10, 10))
    bmask = np.zeros((10, 10), dtype=np.int32)
    bad_flags = 2**0

    wgt = weight.copy()
    wgt[inds] = 0
    assert not slice_full_edge_masked(
        weight=wgt, bmask=bmask, bad_flags=0)

    bm = bmask.copy()
    bm[inds] = 2**0
    assert not slice_full_edge_masked(
        weight=weight, bmask=bm, bad_flags=bad_flags)


def test_compute_masked_fraction():
    weight = np.ones((10, 10))
    bmask = np.zeros((10, 10), dtype=np.int32)
    bad_flags = 2**0

    weight[4, 7] = 0.0
    bmask[7, 9] = 4  # try a different flag
    bmask[8, 2] = 2**0

    assert compute_masked_fraction(
        weight=weight, bmask=bmask, bad_flags=bad_flags) == 0.02


def test_slice_has_flags():
    bmask = np.zeros((10, 10), dtype=np.int32)
    flags = 2**0

    bmask[6, 7] = 4
    assert not slice_has_flags(bmask=bmask, flags=flags)

    bmask[6, 6] = 2**0
    assert slice_has_flags(bmask=bmask, flags=flags)
