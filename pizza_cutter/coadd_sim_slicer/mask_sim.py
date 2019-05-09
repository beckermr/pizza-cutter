import numpy as np
import copy
from ..slice_utils.symmetrize import (
    symmetrize_bmask,
    symmetrize_weight)
from ..slice_utils.interpolate import interpolate_image_and_noise
from ngmix.observation import ObsList, MultiBandObsList


def interpolate_ngmix_multiband_obs(
        *, mbobs, rng, se_interp_flags,
        noise_interp_flags, symmetrize_masking):
    """Interpolate an ngmix multiband observation over bit masked and
    zero weight pixels.

    Parameters
    ----------
    mbobs : `ngmix.observation.MultiBandObsList`
        The input multiband observation to be interpolated.
    rng : `np.random.RandomState`
        An RNG to use for the noise interpolation.
    se_interp_flags : int
        Flags for pixels in the bit mask that will be interpolated via
        a cubic 2D- interpolant.
    noise_interp_flags : int
        Flags for pixels in the bit mask that will be interpolated with noise.
    symmetrize_masking : bool
        If `True`, make the masks symmetric via or-ing them with themselves
        rotated by 90 degrees.

    Returns
    -------
    interp_mbobs : `ngmix.observation.MultiBandObsList`
        The interpolated multiband observation.
    """
    new_mbobs = MultiBandObsList()
    new_mbobs.meta = copy.deepcopy(mbobs.meta)  # safer?
    for obslist in mbobs:
        new_obslist = ObsList()
        new_obslist.meta = copy.deepcopy(obslist.meta)  # safer?

        for obs in obslist:
            (interp_image, sym_weight,
             sym_bmask, interp_noise) = apply_bmask_symmetrize_and_interp(
                se_interp_flags=se_interp_flags,
                noise_interp_flags=noise_interp_flags,
                symmetrize_masking=symmetrize_masking,
                rng=rng,
                image=obs.image.copy(),
                weight=obs.weight.copy(),
                bmask=obs.bmask.copy(),
                noise=obs.noise.copy())

            new_obs = obs.copy()
            new_obs.image = interp_image
            new_obs.weight = sym_weight
            new_obs.bmask = sym_bmask
            new_obs.noise = interp_noise

            new_obslist.append(new_obs)
        new_mbobs.append(new_obslist)

    return new_mbobs


def apply_bmask_symmetrize_and_interp(
        *, se_interp_flags, noise_interp_flags, symmetrize_masking,
        rng, image, weight, bmask, noise):
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
    rng : np.random.RandomState
        RNG instance to use for noise interpolation.
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
        symmetrize_bmask(bmask=bmask)
        symmetrize_weight(weight=weight)

    # first we do the noise interp
    msk = (bmask & noise_interp_flags) != 0
    # we always generate an image to make the RNG use more predictable
    _nse = rng.normal(size=image.shape)
    if np.any(msk):
        zwgt_msk = weight == 0.0
        med_wgt = np.median(weight[~zwgt_msk])
        _nse *= np.sqrt(1.0 / (weight * (~zwgt_msk) + zwgt_msk * med_wgt))
        image[msk] = _nse[msk]

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
