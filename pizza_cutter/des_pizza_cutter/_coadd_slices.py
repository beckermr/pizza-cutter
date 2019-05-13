import logging

import numpy as np

import meds.meds

from ..slice_utils import procflags

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
        mask_tape_bumps,
        edge_buffer,
        wcs_type,
        psf_type,
        rng):
    """Build the inputs to coadd a single slice.

    Parameters
    ----------
    ra : float
        The ra at the center of the coadd patch.
    dec : float
        The dec at the center of the coadd patch.
    ra_psf : float
        The ra where the PSF should be drawn. This should correspond to the
        center of a coadd pixel near the center of a coadd patch.
    dec_psf : float
        The dec where the PSF should be drawn. This should correspond to the
        center of a coadd pixel near the center of a coadd patch.
    box_size : int
        The size of the coadd in pixels.
    se_src_info : list of dicts
        The 'src_info' entry from the `desmeds` info outputs.
    reject_outliers : bool
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    symmetrize_masking : bool
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    coadding_weight : str
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    noise_interp_flags : int
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    se_interp_flags : int
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    bad_image_flags : int
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    max_masked_fraction : float
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    max_unmasked_trail_fraction : float
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    mask_tape_bumps: boold
        If True, turn on TAPEBUMP flag and turn off SUSPECT in bmask
    edge_buffer : int
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    wcs_type : str
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    psf_type : str
        See the documentation of `make_des_pizza_slices` in
        `pizza_cutter.des_pizz_cutter`.
    rng : np.random.RandomState
        An RNG to use in the coadding process.

    Returns
    -------
    slices : list of SEImageSlice objects
        The list of SE images and associated metdata to coadd.
    weights : np.ndarray
        The relative weights to be applied to the SE images when
        coadding.
    slices_not_used : list of SEImageSlice objects
        A list of the slices not used.
    flags_not_used : list of int
        List of flag values for the slices not used.
    """

    # we first do a rough cut of the images
    # this is fast and lets us stop if nothing can be used
    logger.debug('generating slice objects')
    possible_slices = []
    for se_info in se_src_info:
        if se_info['image_flags'] == 0:
            # no flags so init the object
            se_slice = SEImageSlice(
                source_info=se_info,
                psf_model=se_info['%s_psf' % psf_type],
                wcs=se_info['%s_wcs' % wcs_type],
                noise_seed=se_info['noise_seed'],
                mask_tape_bumps=mask_tape_bumps,
            )

            # first try a very rough cut on the patch center
            if se_slice.ccd_contains_radec(ra, dec):

                # if the rough cut worked, then we do a more exact
                # intersection test
                patch_bnds = se_slice.compute_slice_bounds(
                    ra, dec, box_size)
                if se_slice.ccd_contains_bounds(
                        patch_bnds, buffer=edge_buffer):
                    # we found one - set the slice (also does i/o of image
                    # data products)
                    se_slice.set_slice(
                        patch_bnds.colmin,
                        patch_bnds.rowmin,
                        box_size)

                    # we will want this for storing in the meds file,
                    # even if this slice doesn't ultimately get used
                    logger.debug('drawing the PSF')
                    se_slice.set_psf(ra_psf, dec_psf)

                    possible_slices.append(se_slice)

    logger.debug('images found in rough cut: %d', len(possible_slices))

    # just stop now if we find nothing
    if not possible_slices:
        return [], [], [], []

    # we reject outliers after scaling the images to the same zero points
    # every here is passed by reference so this just works
    if reject_outliers:
        logger.debug('rejecting outliers')
        nreject = meds.meds.reject_outliers(
            [s.image for s in possible_slices],
            [s.weight for s in possible_slices])
        logger.debug('# of rejected pixels: %d', nreject)

    # finally, we do the final set of cuts and interpolation
    # any image that passes those gets bad pixels interpolated (slow!)
    # and we call the PSF reconstruction
    slices = []
    weights = []
    slices_not_used = []
    flags_not_used = []
    for se_slice in possible_slices:
        logger.debug('proccseeing image %s', se_slice.source_info['filename'])

        flags = 0

        # the test for interpolating a full edge is done only with the
        # non-noise flags (se_interp_flags)
        # however, we compute the masked fraction using both the noise and
        # interp flags (called bad_flags below)
        # we also symmetrize both
        bad_flags = se_interp_flags | noise_interp_flags

        # this operates in place on references
        if symmetrize_masking:
            logger.debug('symmetrizing the masks')
            symmetrize_bmask(bmask=se_slice.bmask)
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
            msg = []
            if skip_edge_masked:
                msg.append('full edge masked')
                flags |= procflags.FULL_EDGE_MASKED

            if skip_has_flags:
                msg.append('bad image flags')
                flags |= procflags.SLICE_HAS_FLAGS

            if skip_masked_fraction:
                msg.append('masked fraction too high')
                flags |= procflags.HIGH_MASKED_FRAC

            if skip_unmasked_trail_too_big:
                msg.append('unmasked bleed trail too big')
                flags |= procflags.HIGH_UNMASKED_TRAIL_FRAC

            msg = '; '.join(msg)
            logger.debug('rejecting image %s: %s',
                         se_slice.source_info['filename'], msg)

            slices_not_used.append(se_slice)
            flags_not_used.append(flags)
        else:
            # first we do the noise interp
            logger.debug('doing noise interpolation')
            msk = (se_slice.bmask & noise_interp_flags) != 0
            if np.any(msk):
                zmsk = se_slice.weight <= 0.0
                se_wgt = (
                    se_slice.weight * (~zmsk) +
                    np.max(se_slice.weight[~zmsk]) * zmsk)
                noise = (
                    rng.normal(size=se_slice.weight.shape) *
                    np.sqrt(1.0/se_wgt))
                se_slice.image[msk] = noise[msk]

            # now do the cubic interp - note that this will use the noise
            # inteprolated values
            # the same thing is done for the noise field since the noise
            # interpolation above is equivalent to drawing noise
            logger.debug('doing image interpolation')
            interp_image, interp_noise = interpolate_image_and_noise(
                image=se_slice.image,
                noise=se_slice.noise,
                weight=se_slice.weight,
                bmask=se_slice.bmask,
                bad_flags=se_interp_flags,
            )

            if interp_image is None or interp_noise is None:
                flags |= procflags.HIGH_INTERP_MASKED_FRAC
                slices_not_used.append(se_slice)
                flags_not_used.append(flags)

                logger.debug(
                    'rejecting image %s: interpolated region too big',
                    se_slice.source_info['filename'])
                continue

            se_slice.image = interp_image
            se_slice.noise = interp_noise

            slices.append(se_slice)

            weights.append(_build_coadd_weight(
                coadding_weight=coadding_weight,
                weight=se_slice.weight,
                psf=se_slice.psf))

    if weights:
        weights = np.array(weights)
        weights /= np.sum(weights)

    return slices, weights, slices_not_used, flags_not_used


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
        The coadd WCS object.
    wcs_position_offset : int
        The position offset to get from zero-indexed, pixel-centered
        coordinates to the coordinates expected by the coadd WCS object.
    start_row : int
        The starting row/y value of the coadd patch in zero-indexed,
        pixel-centered coordinates.
    start_col : int
        The starting column/x value of the coadd patch in zero-indexed,
        pixel-centered coordinates.
    box_size : int
        The size of the slice in the coadd in pixels.
    psf_start_row : int
        The starting row/y value of the coadd PSF stamp in zero-indexed,
        pixel-centered coordinates. Note that the coadding code draws the PSF
        at a single location in the coadd pixel WCS.
    psf_start_col : int
        The starting column/x value of the coadd PSF stamp in zero-indexed,
        pixel-centered coordinates. Note that the coadding code draws the PSF
        at a single location in the coadd pixel WCS.
    psf_box_size : int
        The size in pixels of the PSF stamp. This number should be odd and big
        enough to fit any of the SE PSF images.
    noise_interp_flags : int
        The SE image flags where noise interpolation has been applied. These
        pixels are mapped to the nearest coadd pixel in the coadd bit mask.
        The coadd bit mask is set to the value `BMASK_NOISE_INTERP` from
        `pizza_cutter.des_pizz_cutter._constants`.
    se_interp_flags : int
        The SE image flags where cubic interpolation has been applied. These
        pixels are mapped to the nearest coadd pixel in the coadd bit mask.
        The coadd bit mask is set to the value `BMASK_SE_INTERP` from
        `pizza_cutter.des_pizz_cutter._constants`.
    se_image_slices : list of SEImageSlice objects
        The list of the SE image slices to be coadded.
    weights : np.ndarray
        An array of weights to apply to each coadd image slice.

    Returns
    -------
    image : np.ndarray
        The coadded images.
    bmask : np.ndarray
        The bit mask for the coadded image. These bit reflect anything
        from the coadd processing.
    ormask : np.ndarray
        An "or" mask of the SE bit mask. Each pixel in the SE bit mask is
        mapped to the nearest coadd pixel and the values are "or"-ed over the
        images in the stack.
    noise : np.ndarray
        The coadded noise field from the SE images.
    psf : np.ndarray
        The coadded PSF model.
    weight : np.ndarray
        The coadded weight map from the SE images.
    """

    # normalize just in case
    weights = np.atleast_1d(weights) / np.sum(weights)

    # make sure input data is consistent
    assert len(se_image_slices) == len(weights), (
        "The input set of weights and images are different sizes.")

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

        # for the PSF, we make sure any NaNs are zero
        msk = ~np.isfinite(resampled_data['psf'])
        if np.any(msk):
            resampled_data['psf'][msk] = 0
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
