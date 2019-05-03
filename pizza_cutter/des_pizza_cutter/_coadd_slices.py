import logging

import numpy as np

import meds.meds

from ._slice_flagging import (
    compute_unmasked_trail_fraction)
from ..slice_utils.flag import (
    slice_full_edge_masked,
    slice_has_flags,
    compute_masked_fraction)
from ..slice_utils.interpolate import interpolate_image_and_noise
from ..slice_utils.symmetrize import (
    symmetrize_bmask,
    symmetrize_weight)
from ..slice_utils.measure import measure_fwhm

from ._se_image import SEImageSlice
from ._constants import BMASK_SE_INTERP, BMASK_NOISE_INTERP

logger = logging.getLogger(__name__)


def _build_coadd_weight(*, coadding_weight, weight, psf):
    # build the weight
    if coadding_weight == 'noise':
        # assume variance is constant
        wt = np.max(weight)
    elif coadding_weight == 'noise-fwhm':
        # assume variance is constant
        wt = np.max(weight)
        fwhm = measure_fwhm(psf)
        wt /= fwhm**4
    else:
        raise ValueError(
            "Coadding weight type '%s' not recognized!" % coadding_weight)

    return wt


def _build_slice_inputs(
        *, ra, dec, ra_psf, dec_psf, box_size,
        coadd_info, start_row, start_col,
        se_src_info,
        reject_outliers,
        symmetrize_masking,
        coadding_weight,
        noise_interp_flags,
        se_interp_flags,
        bad_image_flags,
        max_masked_fraction,
        max_unmasked_trail_fraction,
        rng):
    """Build the inputs to coadd a single slice.

    Parameters
    ----------
    ra : float
    dec : float
    ra_psf : float
    dec_psf : float
    box_size : int
    se_src_info : list of dicts
    reject_outliers : bool
    symmetrize_masking : bool
    coadding_weight : str
    noise_interp_flags : int
    se_interp_flags : int
    bad_image_flags : int
    max_masked_fraction : float
    max_unmasked_trail_fraction : float
    rng : np.random.RandomState

    Returns
    -------
    slices : list of SEImageSlice objects
    weights : np.ndarray
    """

    # we first do a rough cut of the images
    # this is fast and lets us stop if nothing can be used
    logger.debug('generating slice objects')
    seeds = rng.randint(low=1, high=2**30, size=len(se_src_info))
    slices_to_use = []
    for seed, se_info in zip(seeds, se_src_info):
        if se_info['image_flags'] == 0:
            # no flags so init the object
            se_slice = SEImageSlice(
                source_info=se_info,
                psf_model=se_info['piff_psf'],
                wcs=se_info['pixmappy_wcs'],
                noise_seed=seed)

            # first try a very rough cut on the patch center
            if se_slice.ccd_contains_radec(ra, dec):

                # if the rough cut worked, then we do a more exact
                # intersection test
                patch_bnds = se_slice.compute_slice_bounds(
                    ra, dec, box_size)
                if se_slice.ccd_contains_bounds(patch_bnds):
                    # we found one - set the slice (also does i/o of image
                    # data products)
                    se_slice.set_slice(
                        patch_bnds.colmin,
                        patch_bnds.rowmin,
                        box_size)
                    slices_to_use.append(se_slice)
    logger.debug('images found in rough cut: %d', len(slices_to_use))

    # just stop now if we find nothing
    if not slices_to_use:
        return [], []

    # we reject outliers after scaling the images to the same zero points
    # every here is passed by reference so this just works
    if reject_outliers:
        logger.debug('rejecting outliers')
        nreject = meds.meds.reject_outliers(
            [s.image for s in slices_to_use],
            [s.weight for s in slices_to_use])
        logger.debug('# of rejected pixels: %d', nreject)

    # finally, we do the final set of cuts and interpolation
    # any image that passes those gets bad pixels interpolated (slow!)
    # and we call the PSF reconstruction
    slices = []
    weights = []
    for se_slice in slices_to_use:
        logger.debug('proccseeing image %s', se_slice.source_info['filename'])

        # the test for interpolating a full edge is done only with the
        # non-noise flags (se_interp_flags)
        # however, we compute the masked fraction using both the noise and
        # interp flags (called bad_flags below)
        # we also symmetrize both
        bad_flags = se_interp_flags | noise_interp_flags

        # this operates in place on references
        if symmetrize_masking:
            logger.debug('symmetrizing the masks')
            symmetrize_bmask(bmask=se_slice.bmask, bad_flags=bad_flags)
            symmetrize_weight(weight=se_slice.weight)

        skip_edge_masked = slice_full_edge_masked(
                weight=se_slice.weight, bmask=se_slice.bmask,
                bad_flags=se_interp_flags)
        skip_has_flags = slice_has_flags(
            bmask=se_slice.bmask, flags=bad_image_flags)
        skip_masked_fraction = (
            compute_masked_fraction(
                weight=se_slice.weight, bmask=se_slice.bmask,
                bad_flags=bad_flags) >=
            max_masked_fraction)
        skip_unmasked_trail_too_big = compute_unmasked_trail_fraction(
            bmask=se_slice.bmask) >= max_unmasked_trail_fraction

        skip = (
            skip_edge_masked or
            skip_has_flags or
            skip_masked_fraction or
            skip_unmasked_trail_too_big)

        if skip:
            if skip_edge_masked:
                msg = 'full edge masked'
            elif skip_has_flags:
                msg = 'bad image flags'
            elif skip_masked_fraction:
                msg = 'masked fraction too high'
            elif skip_unmasked_trail_too_big:
                msg = 'unmasked bleed trail too big'
            logger.debug('rejecting image %s: %s',
                         se_slice.source_info['filename'], msg)
        else:
            # first we do the noise interp
            logger.debug('doing noise interpolation')
            msk = (se_slice.bmask & noise_interp_flags) != 0
            if np.any(msk):
                noise = (
                    rng.normal(size=se_slice.weight.shape) *
                    np.sqrt(1.0/se_slice.weight))
                se_slice.image[msk] = noise[msk]

            # now do the cubic interp - note that this will use the noise
            # inteprolated values
            # the same thing is done for the noise field since the noise
            # interpolation above is equivalent to drawing noise
            logger.debug('doing image interpolation')
            interp_image, interp_noise = interpolate_image_and_noise(
                image=se_slice.image,
                weight=se_slice.weight,
                bmask=se_slice.bmask,
                bad_flags=se_interp_flags,
                rng=rng)

            if interp_image is None or interp_noise is None:
                logger.debug(
                    'rejecting image %s: interpolated region too big',
                    se_slice.source_info['filename'])
                continue

            se_slice.image = interp_image
            se_slice.noise = interp_noise

            logger.debug('drawing the PSF')
            se_slice.set_psf(ra_psf, dec_psf)

            slices.append(se_slice)

            weights.append(_build_coadd_weight(
                coadding_weight=coadding_weight,
                weight=se_slice.weight,
                psf=se_slice.psf))

    if weights:
        weights = np.array(weights)
        weights /= np.sum(weights)

    return slices, weights


def _coadd_slice_inputs(
        *, wcs, wcs_position_offset, start_row, start_col,
        box_size, psf_start_row, psf_start_col, psf_box_size,
        noise_interp_flags, se_interp_flags,
        se_image_slices, weights):
    """Coadd the slice inputs to form the coadded image, noise realization,
    psf, and weight map.

    Parameters
    ----------
    wcs : `esutil.wcsutil.WCS` object
    wcs_position_offset : int
    start_row : int
    start_col : int
    box_size : int
    psf_start_row : int
    psf_start_col : int
    psf_box_size : int
    noise_interp_flags : int
    se_interp_flags : int
    se_image_slices : list of SEImageSlice objects
    weights : np.ndarray

    Returns
    -------
    image : np.ndarray
    bmask : np.ndarray
    ormask : np.ndarray
    noise : np.ndarray
    psf : np.ndarray
    weight : np.ndarray
    """

    image = np.zeros(
        (box_size, box_size), dtype=se_image_slices[0].image.dtype)
    bmask = np.zeros((box_size, box_size), dtype=np.int32)
    ormask = np.zeros((box_size, box_size), dtype=np.int32)
    noise = np.zeros(
        (box_size, box_size), dtype=se_image_slices[0].image.dtype)
    psf = np.zeros(
        (psf_box_size, psf_box_size), dtype=se_image_slices[0].image.dtype)

    for se_slice, weight in zip(se_image_slices, weights):
        logger.debug('resampling image %s', se_slice.source_info['filename'])
        resampled_data = se_slice.resample(
            wcs=wcs,
            wcs_position_offset=wcs_position_offset,
            x_start=start_col,
            y_start=start_row,
            box_size=box_size,
            psf_x_start=psf_start_col,
            psf_y_start=psf_start_row,
            psf_box_size=psf_box_size
        )

        image += (resampled_data['image'] * weight)
        noise += (resampled_data['noise'] * weight)
        psf += (resampled_data['psf'] * weight)

        ormask |= resampled_data['bmask']

        msk = (resampled_data['bmask'] & noise_interp_flags) != 0
        bmask[msk] |= BMASK_NOISE_INTERP

        msk = (resampled_data['bmask'] & se_interp_flags) != 0
        bmask[msk] |= BMASK_SE_INTERP

    weight = np.zeros_like(noise)
    weight[:, :] = 1.0 / np.var(noise)

    return (
        image,
        bmask,
        ormask,
        noise,
        psf,
        weight)
