import numpy as np
from ..slice_utils.symmetrize import (
    symmetrize_bmask,
    symmetrize_weight)
from ..slice_utils.interpolate import interpolate_image_and_noise


def apply_bmask_symmetrize_and_interp(
        *, se_interp_flags, noise_interp_flags, symmetrize_masking,
        noise_for_noise_interp, image, weight, bmask, noise):
    """Mask a coadd sim in a way that approximates what is done with the
    real data.

    Parameters
    ----------
    se_interp_flags : int
        Flags for pixels in the bit mask that will be interpolated via
        a cubic 2D- interpolant.
    noise_interp_flags : int
        Flags for pixels in the bit mask that will be interpolated with noise.
    symmetrize_masking : bool
        If `True`, make the masks symmetric via or-ing them with themselves
        rotated by 90 degrees.
    noise_for_noise_interp : array-like
        An array of noise to be used for the noise interpolation.
    image : array-like
        The image to process.
    weight : array-like
        The weight map of the image to process.
    bmask : array-like
        The bit mask of the image to process.
    noise : array-like
        The noise field of the image to process. This noise field will be
        interpolated as well.

    Returns
    -------
    interp_image : array-like
        The intrpolated image.
    weight : array-like
        The symmetrized weight map.
    bmask : array-like
        The symmetrized bit mask.
    interp_noise : array-like
        The interpolated noise field.
    """
    # maybe symmetrize the bmask and weight
    if symmetrize_masking:
        bad_flags = se_interp_flags | noise_interp_flags
        symmetrize_bmask(bmask=bmask, bad_flags=bad_flags)
        symmetrize_weight(weight=weight)

    # first we do the noise interp
    msk = (bmask & noise_interp_flags) != 0
    if np.any(msk):
        image[msk] = noise_for_noise_interp[msk]

    # now do the cubic interp - note that this will use the noise
    # inteprolated values
    # the same thing is done for the noise field since the noise
    # interpolation above is equivalent to drawing noise
    interp_image, interp_noise = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=se_interp_flags,
        rng=None,
        noise=noise)

    return interp_image, weight, bmask, interp_noise
