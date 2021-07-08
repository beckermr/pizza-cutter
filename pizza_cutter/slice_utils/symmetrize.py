import numpy as np
import scipy.ndimage


def symmetrize_weight(*, weight, angle=None):
    """Symmetrize zero weight pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    weight : array-like
        The weight map for the slice.
    angle : float
        If not None, then the routine `scipy.ndimage.rotate` is used for the mask
        symmetrization with this angle in degrees. Note that specifying 90 degress
        will not produce the same output as specifying None.
    """
    if weight.shape[0] != weight.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    if angle is None:
        weight_rot = np.rot90(weight)
        msk = weight_rot == 0.0
        if np.any(msk):
            weight[msk] = 0.0
    else:
        msk = weight == 0.0
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


def symmetrize_bmask(*, bmask, angle=None):
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
    """
    if bmask.shape[0] != bmask.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    if angle is None:
        bmask |= np.rot90(bmask)
    else:
        for i in range(32):
            bit = 2**i
            msk = (bmask & bit) != 0
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
