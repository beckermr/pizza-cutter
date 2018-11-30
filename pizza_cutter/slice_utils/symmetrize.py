import numpy as np


def symmetrize_weight(*, weight):
    """Symmetrize zero weight pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    weight : array-like
        The weight map for the slice.
    """
    if weight.shape[0] != weight.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    weight_rot = np.rot90(weight)
    msk = weight_rot == 0.0
    if np.any(msk):
        weight[msk] = 0.0


def symmetrize_bmask(*, bmask, bad_flags):
    """Symmetrize masked pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        The flags to symmetrize in the bit mask using
        `(bmask & bad_flags) != 0`.
    """
    if bmask.shape[0] != bmask.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    bm_rot = np.rot90(bmask)
    msk = (bm_rot & bad_flags) != 0
    if np.any(msk):
        bmask[msk] |= bm_rot[msk]
