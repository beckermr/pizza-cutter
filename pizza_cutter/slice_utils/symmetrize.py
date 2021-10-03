import numpy as np
import scipy.ndimage


def symmetrize_weight(*, weight, angle=None, no_sym_mask=None):
    """Symmetrize zero weight pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    weight : array-like
        The weight map for the slice.
    angle : float, optional
        If not None, then the routine `scipy.ndimage.rotate` is used for the mask
        symmetrization with this angle in degrees. Note that specifying 90 degress
        will not produce the same output as specifying None.
    no_sym_mask : np.ndarray, optional
        If passed, only zero-weight pixels at locations where this mask is False will be
        symmetrized.
    """
    if weight.shape[0] != weight.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    if no_sym_mask is None:
        mask = np.ones_like(weight).astype(bool)
    else:
        mask = ~no_sym_mask

    if angle is None:
        mask = np.rot90(mask)
        weight_rot = np.rot90(weight)
        msk = (weight_rot == 0.0) & mask
        if np.any(msk):
            weight[msk] = 0.0
    else:
        msk = (weight == 0.0) & mask
        if np.any(msk):
            rot_wgt = np.zeros_like(weight)
            rot_wgt[msk] = 1.0
            rot_wgt = scipy.ndimage.rotate(
                rot_wgt,
                angle,
                reshape=False,
                order=1,
                mode='constant',
                cval=1.0,
            )
            msk = rot_wgt > 0
            if np.any(msk):
                weight[msk] = 0.0


def symmetrize_bmask(*, bmask, angle=None, no_sym_mask=None, sym_flags=None):
    """Symmetrize masked pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    bmask : array-like
        The bit mask for the slice.
    angle : float
        If not None, then the routine `scipy.ndimage.rotate` is used for the mask
        symmetrization with this angle in degrees. Note that specifying 90 degress
        will not produce the same output as specifying None.
    no_sym_mask : np.ndarray, optional
        If passed, only bmask pixels at locations where this mask is False will be
        symmetrized.
    sym_flags : int, optional
        If specified, symmetrize only these flags. Otherwise symmetrize all of them.
    """
    if bmask.shape[0] != bmask.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    if angle is None:
        if no_sym_mask is not None:
            # if we are not symmetrizing a given pixel, we set its flags to zero
            # before we apply a rotation
            bmask_keep = bmask.copy()
            bmask_keep[no_sym_mask] = 0
        else:
            bmask_keep = bmask

        if sym_flags is not None:
            bmask |= np.rot90(bmask_keep & sym_flags)
        else:
            bmask |= np.rot90(bmask_keep)
    else:
        if no_sym_mask is not None:
            mask = ~no_sym_mask

        for i in range(32):
            bit = 2**i
            if sym_flags is not None and ((bit & sym_flags) == 0):
                continue

            msk = (bmask & bit) != 0
            if no_sym_mask is not None:
                msk &= mask

            if np.any(msk):
                mask_for_bit = np.zeros_like(bmask, dtype=np.float32)
                mask_for_bit[msk] = 1.0
                mask_for_bit = scipy.ndimage.rotate(
                    mask_for_bit,
                    angle,
                    reshape=False,
                    order=1,
                    mode='constant',
                    cval=1.0,
                )
                msk = mask_for_bit > 0
                bmask[msk] |= bit
