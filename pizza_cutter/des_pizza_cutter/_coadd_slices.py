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
from ..slice_utils.interpolate import (
    interpolate_image_and_noise,
    copy_masked_edges_image_and_noise,
)
from ..slice_utils.symmetrize import (
    symmetrize_bmask,
    symmetrize_weight)
from ..slice_utils.measure import measure_fwhm

from ._se_image import SEImageSlice
from ._constants import BMASK_SPLINE_INTERP, BMASK_NOISE_INTERP

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
        frac_buffer,
        coadd_info, start_row, start_col,
        se_src_info,
        rng,
        coadding_weight,
        reject_outliers,
        symmetrize_masking,
        copy_masked_edges,
        noise_interp_flags,
        spline_interp_flags,
        bad_image_flags,
        max_masked_fraction,
        max_unmasked_trail_fraction,
        mask_tape_bumps,
        edge_buffer,
        wcs_type,
        psf_type,
        tmpdir=None):
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
    frac_buffer : float
        The fractional amount by which to increse the coadd box size. Set
        up to sqrt(2) to account for full position angle rotations. In DES
        this number should be very tiny or zero.
    se_src_info : list of dicts
        The 'src_info' entry from the info file.
    rng : np.random.RandomState
        An RNG to use in the coadding process.
    coadding_weight : str
        The kind of relative weight to apply to each of the SE images that
        form a coadd. The options are

        'noise' - use the maximum of the weight map for each SE image.
        'noise-fwhm' - use the maximum of the weight map divided by the
            (PSF FWHM)**4
    tmpdir: optional, string
        Optional temporary directory for temporary files

    These next options are from the 'single_epoch' section of the main
    configuration file (passed around as `single_epoch_config` in the code).

    reject_outliers : bool
        If True, assume the SE images are approximatly registered with
        respect to one another and apply the pixel outlier rejection
        code from the `meds` package. If False, this step is skipped.
    symmetrize_masking : bool
        If True, the bit masks and any zero weight pixels will be rotated
        by 90 degrees and applied again to the weight and bit masks. If False,
        this step is skipped.
    copy_masked_edges : bool
        If True, copy pixels from adjacent column or row into masked edge pixels
        if the slice edge is fully masked.
    noise_interp_flags : int
        An "or" of bit flags. Any pixel in the image with one or more of these
        flags will be replaced by noise (computed from the weight map) before
        coadding. This step is done after any mask symmetrization.
    spline_interp_flags : int
        An "or" of bit flags. Any pixel in the image with one or more of these
        flags will be interpolated using a cubic order interpolant over the
        good pixels. This step is done after symmetrization of the mask and
        any noise interpolation via `noise_interp_flags`.
    bad_image_flags : int
        An "or" of bit flags. Any image the set of SE images with any pixel
        in the coadding region set to one of these flags is ignored during
        coadding.
    max_masked_fraction : float
        The maximum masked fraction an SE image can have before it is
        excluded from the coadd. This masked fraction is computed from any
        zero weight pixels, any picels with any of the se_interp_flags or
        any pixels with any of the noise_interp_flags. It is the fraction of
        the subset of the SE image that approximatly overlaps the final coadded
        region.
    max_unmasked_trail_fraction : float
        The maximum unmasked bleed trail fraction an SE image can have
        before it is exlcuded from the coadd. This parameter is the
        fraction of the subset of the SE image that overlaps the coadd. See
        the function `compute_unmasked_trail_fraction` in
        `pizza_cutter.des_pizz_cutter._slice_flagging.` for details.
    mask_tape_bumps: bool
        If True, turn on TAPEBUMP flag and turn off SUSPECT in bmask. This
        option is only applicable to DES Y3 processing.
    edge_buffer : int
        A buffer region of this many pixels will be excluded from the coadds.
        Note that any SE image whose relevant region for a given coadd
        intersects with this region will be fully excluded from the coadd,
        even if it has area that could be used.
    wcs_type : str
        The SE WCS solution to use for coadd. This should be one of 'pixmappy'
        or 'scamp'.
    psf_type : str
        The SE PSF model to use. This should be one 'psfex' or 'piff'.

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
    logger.info('generating slice objects for ra,dec = %s|%s', ra, dec)
    possible_slices = []
    for se_info in se_src_info:
        if se_info['image_flags'] == 0:
            # no flags so init the object
            se_slice = SEImageSlice(
                source_info=se_info,
                psf_model=se_info['%s_psf' % psf_type],
                wcs=se_info['%s_wcs' % wcs_type],
                wcs_position_offset=se_info['position_offset'],
                noise_seed=se_info['noise_seed'],
                mask_tape_bumps=mask_tape_bumps,
                tmpdir=tmpdir,
            )

            # first try a very rough cut on the patch center
            if se_slice.ccd_contains_radec(ra, dec):

                # if the rough cut worked, then we do a more exact
                # intersection test
                patch_bnds = se_slice.compute_slice_bounds(
                    ra, dec, box_size, frac_buffer)
                if se_slice.ccd_contains_bounds(
                    patch_bnds, buffer=edge_buffer
                ):
                    logger.debug(
                        'found possible image %s/%s',
                        se_slice.source_info['path'],
                        se_slice.source_info['filename'],
                    )

                    # we found one - set the slice (also does i/o of image
                    # data products)
                    se_slice.set_slice(patch_bnds)

                    # we will want this for storing in the meds file,
                    # even if this slice doesn't ultimately get used
                    se_slice.set_psf(ra_psf, dec_psf)

                    possible_slices.append(se_slice)

    logger.info('images found in rough cut: %d', len(possible_slices))

    # just stop now if we find nothing
    if not possible_slices:
        return [], [], [], []

    # we reject outliers after scaling the images to the same zero points
    # every here is passed by reference so this just works
    if reject_outliers:
        logger.info('rejecting outliers')
        nreject = meds.meds.reject_outliers(
            [s.image for s in possible_slices],
            [s.weight for s in possible_slices])
        logger.info('# of rejected pixels: %d', nreject)

    # finally, we do the final set of cuts and interpolation
    # any image that passes those gets bad pixels interpolated (slow!)
    # and we call the PSF reconstruction
    slices = []
    weights = []
    slices_not_used = []
    flags_not_used = []
    for se_slice in possible_slices:
        logger.info(
            'pre-proccseeing image %s/%s',
            se_slice.source_info["path"],
            se_slice.source_info['filename'],
        )

        flags = 0

        # the test for interpolating a full edge is done only with the
        # non-noise flags (se_interp_flags)
        # however, we compute the masked fraction using both the noise and
        # interp flags (called bad_flags below)
        # we also symmetrize both
        bad_flags = spline_interp_flags | noise_interp_flags

        # this operates in place on references
        if symmetrize_masking:
            logger.debug('symmetrizing the masks')
            symmetrize_bmask(bmask=se_slice.bmask)
            symmetrize_weight(weight=se_slice.weight)

        if not copy_masked_edges:
            skip_edge_masked = slice_full_edge_masked(
                    weight=se_slice.weight, bmask=se_slice.bmask,
                    bad_flags=spline_interp_flags)
        else:
            skip_edge_masked = False
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
            logger.info(
                'rejecting image %s: %s',
                se_slice.source_info['filename'],
                msg,
            )

            slices_not_used.append(se_slice)
            flags_not_used.append(flags)
        else:
            interp_image = se_slice.image.copy()
            interp_noise = se_slice.noise.copy()
            interp_weight = se_slice.weight.copy()
            interp_bmask = se_slice.bmask.copy()

            # first we do the noise interp
            msk = (se_slice.bmask & noise_interp_flags) != 0
            if np.any(msk):
                logger.debug('doing noise interpolation')
                zmsk = se_slice.weight <= 0.0
                se_wgt = (
                    se_slice.weight * (~zmsk) +
                    np.max(se_slice.weight[~zmsk]) * zmsk)
                noise = (
                    rng.normal(size=se_slice.weight.shape) *
                    np.sqrt(1.0/se_wgt))
                interp_image[msk] = noise[msk]

                noise = (
                    rng.normal(size=se_slice.weight.shape) *
                    np.sqrt(1.0/se_wgt))
                interp_noise[msk] = noise[msk]

            # then deal with fully masked edges as a special case
            if copy_masked_edges:
                interp_image, interp_noise, interp_bmask, interp_weight \
                    = copy_masked_edges_image_and_noise(
                        image=interp_image,
                        noise=interp_noise,
                        weight=interp_weight,
                        bmask=interp_bmask,
                        bad_flags=spline_interp_flags,
                    )

                # recheck edge masking here just in case
                skip_edge_masked = slice_full_edge_masked(
                        weight=interp_weight, bmask=interp_bmask,
                        bad_flags=spline_interp_flags)

                if skip_edge_masked:
                    flags |= procflags.FULL_EDGE_MASKED
                    slices_not_used.append(se_slice)
                    flags_not_used.append(flags)

                    logger.info(
                        'rejecting image %s: could not correct '
                        'fully masked edges with copy of adjcent pixels',
                        se_slice.source_info['filename']
                    )
                    continue

            # now do the cubic interp - note that this will use the noise
            # inteprolated values
            # the same thing is done for the noise field since the noise
            # interpolation above is equivalent to drawing noise
            interp_image, interp_noise = interpolate_image_and_noise(
                image=interp_image,
                noise=interp_noise,
                weight=interp_weight,
                bmask=interp_bmask,
                bad_flags=spline_interp_flags,
            )

            if interp_image is None or interp_noise is None:
                flags |= procflags.HIGH_INTERP_MASKED_FRAC
                slices_not_used.append(se_slice)
                flags_not_used.append(flags)

                logger.info(
                    'rejecting image %s: interpolated region too big',
                    se_slice.source_info['filename'])
                continue

            # make an image processing mask and set it
            # note we have to make sure this is int32 to get all of the flags
            pmask = np.zeros(se_slice.bmask.shape, dtype=np.int32)

            msk = (se_slice.bmask & noise_interp_flags) != 0
            pmask[msk] |= BMASK_NOISE_INTERP

            msk = (
                (se_slice.weight <= 0) |
                ((se_slice.bmask & spline_interp_flags) != 0))
            pmask[msk] |= BMASK_SPLINE_INTERP

            # set interpolated images
            # we keep original weight and bmask since those were edited only
            # to deal with masked edges
            se_slice.set_interp_image_noise_pmask(
                mask=pmask,
                interp_image=interp_image,
                interp_noise=interp_noise,
            )

            slices.append(se_slice)

            weights.append(_build_coadd_weight(
                coadding_weight=coadding_weight,
                weight=se_slice.weight,
                psf=se_slice.psf))

    if weights:
        weights = np.array(weights)
        weights /= np.sum(weights)
        logger.debug("coadding weights: %s", list(weights))

    return slices, weights, slices_not_used, flags_not_used


def _coadd_slice_inputs(
    *, wcs, wcs_position_offset, wcs_image_shape, start_row, start_col,
    box_size, psf_start_row, psf_start_col, psf_box_size,
    se_image_slices, weights, se_wcs_interp_delta, coadd_wcs_interp_delta,
):
    """Coadd the slice inputs to form the coadded image, noise realization,
    psf, and weight map.

    Parameters
    ----------
    wcs : `esutil.wcsutil.WCS` or `AffineWCS` object
        The coadd WCS object.
    wcs_position_offset : int
        The position offset to get from zero-indexed, pixel-centered
        coordinates to the coordinates expected by the coadd WCS object.
    wcs_image_shape : tuple of ints
        The shape of the full coadd WCS image.
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
    se_image_slices : list of SEImageSlice objects
        The list of the SE image slices to be coadded.
    weights : np.ndarray
        An array of weights to apply to each coadd image slice.
    se_wcs_interp_delta : int
        The spacing in pixels used to interpolate coadd pixel to SE pixel WCS
        function.
    coadd_wcs_interp_delta : int
        The spacing in pixels used for interpolating the coadd WCS pixel area.

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
    interp_se_frac : np.ndarray
        The fraction of interpolated SE images for each pixel. This quantity is
        not flux-weighted (i.e. it is either 0 or 1 for each SE image).
    resampled_data : list of dict
        A list containing a dictionary of the raw resampled data for each SE slice.
    """

    # normalize just in case
    weights = np.atleast_1d(weights) / np.sum(weights)

    # make sure input data is consistent
    assert len(se_image_slices) == len(weights), (
        "The input set of weights and images are different sizes.")

    image = np.zeros(
        (box_size, box_size), dtype=se_image_slices[0].image.dtype)
    interp_se_frac = np.zeros(
        (box_size, box_size), dtype=se_image_slices[0].image.dtype)
    bmask = np.zeros((box_size, box_size), dtype=np.int32)
    ormask = np.zeros((box_size, box_size), dtype=np.int32)
    noise = np.zeros(
        (box_size, box_size), dtype=se_image_slices[0].image.dtype)
    psf = np.zeros(
        (psf_box_size, psf_box_size), dtype=se_image_slices[0].image.dtype)

    resampled_datas = []

    for se_slice, weight in zip(se_image_slices, weights):
        logger.info(
            'resampling image %s/%s',
            se_slice.source_info["path"],
            se_slice.source_info['filename'],
        )
        resampled_data = se_slice.resample(
            wcs=wcs,
            wcs_position_offset=wcs_position_offset,
            wcs_interp_shape=wcs_image_shape,
            x_start=start_col,
            y_start=start_row,
            box_size=box_size,
            psf_x_start=psf_start_col,
            psf_y_start=psf_start_row,
            psf_box_size=psf_box_size,
            se_wcs_interp_delta=se_wcs_interp_delta,
            coadd_wcs_interp_delta=coadd_wcs_interp_delta,
        )
        resampled_datas.append(resampled_data)

        if np.all(resampled_data['image'] == 0):
            logger.warning(
                "resampled image %s/%s is all zero!",
                se_slice.source_info["path"],
                se_slice.source_info["filename"],
            )

        image += (resampled_data['image'] * weight)
        noise += (resampled_data['noise'] * weight)
        interp_se_frac += (resampled_data['interp_frac'] * weight)

        # for the PSF, we make sure any NaNs are zero
        msk = ~np.isfinite(resampled_data['psf'])
        if np.any(msk):
            resampled_data['psf'][msk] = 0
        psf += (resampled_data['psf'] * weight)

        ormask |= resampled_data['bmask']
        bmask |= resampled_data['pmask']

    weight = np.zeros_like(noise)
    weight[:, :] = 1.0 / np.var(noise)

    return (
        image,
        bmask,
        ormask,
        noise,
        psf,
        weight,
        interp_se_frac,
        resampled_datas,
    )
