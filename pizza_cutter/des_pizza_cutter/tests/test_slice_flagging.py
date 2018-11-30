import numpy as np

from .._slice_flagging import (
    compute_unmasked_trail_fraction)


def test_compute_unmasked_trail_fraction():
    bmask = np.zeros((10, 10), dtype=np.int32)
    bmask[0, :] = 64
    bmask[5, 8] = 64
    bmask[0, 5:] |= 32

    assert compute_unmasked_trail_fraction(bmask=bmask) == 0.06
