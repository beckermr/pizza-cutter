import functools
import logging

import scipy.special
import numpy as np

import galsim
import fitsio

import meds.meds
from meds.bounds import Bounds
from meds.util import radec_to_uv

from ._slice_flagging import (
    slice_full_edge_masked,
    slice_has_flags,
    compute_masked_fraction)
from ._slice_data import (
    symmetrize_bmask,
    symmetrize_weight,
    interpolate_image_and_noise)

from ._constants import BMASK_SE_INTERP, BMASK_NOISE_INTERP, BMASK_EDGE

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16)
def _read_image(path, ext):
    """Cached reads of images.

    Each SE image in DES is ~33 MB in float. Thus we use at most ~0.5 GB of
    memory with a 16 element cache.
    """
    return fitsio.read(path, ext=ext)


def _do_rough_cut_and_maybe_extract(
        *, ra, dec, ra_ccd, dec_ccd, sky_bnds, ccd_bnds, wcs,
        half_box_size, box_size):
    """Do a rough check if a CCD overlaps a slice. If yes, return information
    about the proper patch of the CCD to extract. If no, then return an empty
    dictionary.

    The returned dict, if not empty, has the following keys:

        'start_row' : the starting row of the patch
        'start_col' : the starting col of the patch
        'row' : the row location of the slice center (ra, dec)
        'col' : the col location of the slice center (ra, dec)
        'row_offset' : the offset in pixels between the canonical patch
            center row and the slice center row
        'col_offset' : the offset in pixels between the canonical patch
            center col and the slice center col
    """
    ii = {}

    u, v = radec_to_uv(ra, dec, ra_ccd, dec_ccd)
    rough_msk = sky_bnds.contains_points(u, v)

    if rough_msk:
        col, row = wcs.sky2image(longitude=ra, latitude=dec)
        col -= 1
        row -= 1

        se_start_row = np.int32(row) - half_box_size + 1
        se_start_col = np.int32(col) - half_box_size + 1
        se_row_cen = (box_size - 1) / 2 + se_start_row
        se_col_cen = (box_size - 1) / 2 + se_start_col
        se_row_offset = row - se_row_cen
        se_col_offset = col - se_col_cen
        patch_bnds = Bounds(
            se_start_row, se_start_row+box_size,
            se_start_col, se_start_col+box_size)

        if ccd_bnds.contains_bounds(patch_bnds):
            ii = {
                'start_row': se_start_row,
                'start_col': se_start_col,
                'row_offset': se_row_offset,
                'col_offset': se_col_offset,
                'row': row,
                'col': col}

    return ii


def _read_image_lists(*, images_to_use, se_src_info, box_size):
    imlist = []
    wtlist = []
    bmlist = []
    for i, image_info in images_to_use.items():
        src_info = se_src_info[i]

        image = (
            _read_image(src_info['image_path'], ext=src_info['image_ext']) -
            _read_image(src_info['bkg_path'], ext=src_info['bkg_ext']))
        weight = _read_image(
            src_info['weight_path'], ext=src_info['weight_ext'])
        bmask = _read_image(src_info['bmask_path'], ext=src_info['bmask_ext'])

        row = image_info['start_row']
        col = image_info['start_col']

        # we have to make sure to return copies here - otherwise we get a
        # view and then further downstream stuff can muck with things
        imlist.append(
            image[row:row+box_size, col:col+box_size] * src_info['scale'])
        wtlist.append(
            weight[row:row+box_size, col:col+box_size] / src_info['scale']**2)
        bmlist.append(
            bmask[row:row+box_size, col:col+box_size].copy())

    return imlist, wtlist, bmlist


def _build_gs_image(
        *, array, galsim_wcs, start_col, start_row,
        col_offset, row_offset, position_offset, coadding_interp):
    return galsim.InterpolatedImage(
        galsim.ImageD(
            array,
            wcs=galsim_wcs,
            xmin=start_col + position_offset,
            ymin=start_row + position_offset),
        offset=galsim.PositionD(
            x=col_offset, y=row_offset),
        x_interpolant=coadding_interp)


def measure_fwhm(image, smooth=0.1):
    """Measure the image FWHM.

    Parameters
    ----------
    image : 2-d array
        The image to measure.
    smooth : float
        The smoothing scale for the erf. This should be between 0 and 1. If
        you have noisy data, you might set this to the noise value or greater,
        scaled by the max value in the images.  Otherwise just make sure it
        smooths enough to avoid pixelization effects.

    Returns
    -------
    fwhm : float
        The FWHM in pixels.
    """

    thresh = 0.5
    nim = image.copy()
    maxval = image.max()
    nim /= maxval

    arg = (nim - thresh)/smooth

    vals = 0.5 * (1 + scipy.special.erf(arg))

    area = np.sum(vals)
    width = 2 * np.sqrt(area / np.pi)

    return width


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


def _build_psf_image(
        *, row, col, psf_rec, galsim_wcs, position_offset, coadding_interp):
    # we get the PSF image from the nearest position on the pixel grid
    # this is so that we can apply the WCS when doing the coadding w/ galsim
    psf_row = int(row + 0.5)
    psf_col = int(col + 0.5)

    # now reconstruct and normalize so the sum of the pixels is 1
    # TODO: Should this be the integral in sky coords?
    psf = psf_rec.get_rec(psf_row, psf_col)
    assert psf.shape[0] == psf.shape[1]
    psf /= np.sum(psf)

    # now we compute the lower-left corner of the PSF image
    psf_img_center = (psf.shape[0] - 1) / 2
    psf_start_row = psf_row - psf_img_center
    psf_start_col = psf_col - psf_img_center
    # FIXME: this might fail - if it does...crap?
    assert int(psf_start_col) == psf_start_col
    assert int(psf_start_row) == psf_start_row

    # finally, we allow for the center of the PSF to be offset from
    # the canonical image center
    psf_center = psf_rec.get_center(psf_row, psf_col)
    psf_row_offset = psf_center[0] - psf_img_center
    psf_col_offset = psf_center[1] - psf_img_center

    # and do the thing...
    gs_psf = galsim.InterpolatedImage(
        galsim.ImageD(
            psf,
            wcs=galsim_wcs,
            xmin=psf_start_col + position_offset,
            ymin=psf_start_row + position_offset),
        offset=galsim.PositionD(
            x=psf_col_offset, y=psf_row_offset),
        x_interpolant=coadding_interp)

    return psf, gs_psf


def _interpolate_bmask(
        *, box_size, se_bmask,
        se_start_row, se_start_col, se_wcs, se_position_offset,
        start_row, start_col, coadd_wcs, coadd_position_offset):
    """Interpolate a bit mask from an SE exposure to a coadd.
    """

    # first convert row, col to ra, dec
    row, col = np.mgrid[0:box_size, 0:box_size]
    row = row.ravel()
    col = col.ravel()
    ra, dec = coadd_wcs.image2sky(
        x=col + np.int64(start_col) + coadd_position_offset,
        y=row + np.int64(start_row) + coadd_position_offset)

    # then for each SE image, find the row, col, clip to the box, and set
    # values
    ncol, nrow = se_wcs.get_naxis()

    se_col, se_row = se_wcs.sky2image(
        longitude=ra,
        latitude=dec,
        find=False)
    se_col -= se_position_offset
    se_row -= se_position_offset
    se_col = np.clip((se_col + 0.5).astype(np.int64), 0, ncol - 1)
    se_row = np.clip((se_row + 0.5).astype(np.int64), 0, nrow - 1)
    se_col -= se_start_col
    se_row -= se_start_row
    msk = (
        (se_row >= 0) & (se_row < se_bmask.shape[0]) &
        (se_col >= 0) & (se_col < se_bmask.shape[1]))

    coadd_bmask = np.zeros((box_size, box_size), dtype=np.int32)
    coadd_bmask[row[msk], col[msk]] = se_bmask[se_row[msk], se_col[msk]]
    coadd_bmask[row[~msk], col[~msk]] = BMASK_EDGE

    return coadd_bmask


def _build_slice_inputs(
        *, ra, dec, box_size,
        coadd_info, start_row, start_col,
        se_src_info,
        reject_outliers,
        symmetrize_masking,
        coadding_weight,
        coadding_interp,
        noise_interp_flags,
        se_interp_flags,
        bad_image_flags,
        max_masked_fraction,
        rng):
    """Build the inputs to coadd a single slice.

    Parameters
    ----------
    ra : float
    dec : float
    box_size : int
    se_src_info : list of dicts
    reject_outliers : bool
    symmetrize_masking : bool
    coadding_weight : str
    coadding_interp : str
    noise_interp_flags : int
    se_interp_flags : int
    bad_image_flags : int
    max_masked_fraction : float
    rng : np.random.RandomState

    Returns
    -------
    images
    bmasks
    noises
    psfs
    weights
    se_images_kept
    """

    half_box_size = int(box_size / 2)

    # we first do a rough cut of the images
    # this is fast and lets us stop if nothing can be used
    images_to_use = {}
    for i, se_info in enumerate(se_src_info):
        image_info = _do_rough_cut_and_maybe_extract(
            ra=ra,
            dec=dec,
            ra_ccd=se_info['ra_ccd'],
            dec_ccd=se_info['dec_ccd'],
            sky_bnds=se_info['sky_bnds'],
            ccd_bnds=se_info['ccd_bnds'],
            wcs=se_info['wcs'],
            half_box_size=half_box_size,
            box_size=box_size)
        if image_info:
            images_to_use[i] = image_info
    logger.debug('images found in rough cut: %d', len(images_to_use))

    # just stop now if we find nothing
    if not images_to_use:
        return [], [], [], [], [], []

    # we found some stuff - let's read it in
    # note that `_read_image` (called by `_read_image_lists`) is cached
    # to help avoid duplicate i/o
    # this function also scales the images and weight maps to the right
    # zero points
    imlist, wtlist, bmlist = _read_image_lists(
        images_to_use=images_to_use,
        se_src_info=se_src_info,
        box_size=box_size)

    # we reject outliers after scaling the images to the same zero points
    if reject_outliers:
        nreject = meds.meds.reject_outliers(imlist, wtlist)
        logger.debug('# of rejected pixels: %d', nreject)

    # finally, we do the final set of cuts, interp, and build the needed
    # galsim objects
    # any image that passes those gets bad pixels interpolated (slow!)
    # and we call the PSF reconstruction
    gs_images = []
    bmasks = []
    gs_noises = []
    gs_psfs = []
    weights = []
    se_images_kept = []
    for image, weight, bmask, (index, image_info) in zip(
            imlist, wtlist, bmlist, images_to_use.items()):
        logger.debug('proccseeing image %d', index)

        src_info = se_src_info[index]

        # the test for interpolating a full edge is done only with the
        # non-noise flags (se_interp_flags)
        # however, we compute the masked fraction using both the noise and
        # interp flags (called bad_flags below)
        # we also symmetrize both
        bad_flags = se_interp_flags | noise_interp_flags

        if symmetrize_masking:
            symmetrize_bmask(bmask=bmask, bad_flags=bad_flags)
            symmetrize_weight(weight=weight)

        skip_edge_masked = slice_full_edge_masked(
                weight=weight, bmask=bmask, bad_flags=se_interp_flags)
        skip_has_flags = slice_has_flags(bmask=bmask, flags=bad_image_flags)
        skip_masked_fraction = (
            compute_masked_fraction(
                weight=weight, bmask=bmask, bad_flags=bad_flags) >
            max_masked_fraction)

        skip = skip_edge_masked or skip_has_flags or skip_masked_fraction

        if skip:
            if skip_edge_masked:
                msg = 'full edge masked'
            elif skip_has_flags:
                msg = 'bad image flags'
            elif skip_masked_fraction:
                msg = 'masked fraction too high'
            logger.debug('rejecting image %d: %s', index, msg)
        else:
            se_images_kept.append(index)

            # first we do the noise interp
            msk = (bmask & noise_interp_flags) != 0
            if np.any(msk):
                noise = rng.normal(size=weight.shape) * np.sqrt(1.0/weight)
                image[msk] = noise[msk]

            # now do the cubic interp - note that this will use the noise
            # inteprolated values
            # the same thing is done for the noise field since the noise
            # interpolation above is equivalent to drawing noise
            interp_image, interp_noise = interpolate_image_and_noise(
                image=image,
                weight=weight,
                bmask=bmask,
                bad_flags=se_interp_flags,
                rng=rng)

            # finally we build the galsim inteprolated images to be used for
            # coadding
            gs_images.append(_build_gs_image(
                array=interp_image,
                galsim_wcs=src_info['galsim_wcs'],
                start_col=image_info['start_col'],
                start_row=image_info['start_row'],
                col_offset=image_info['col_offset'],
                row_offset=image_info['row_offset'],
                position_offset=src_info['position_offset'],
                coadding_interp=coadding_interp))

            gs_noises.append(_build_gs_image(
                array=interp_noise,
                galsim_wcs=src_info['galsim_wcs'],
                start_col=image_info['start_col'],
                start_row=image_info['start_row'],
                col_offset=image_info['col_offset'],
                row_offset=image_info['row_offset'],
                position_offset=src_info['position_offset'],
                coadding_interp=coadding_interp))

            bmasks.append(_interpolate_bmask(
                box_size=box_size,
                se_bmask=bmask,
                se_start_row=image_info['start_row'],
                se_start_col=image_info['start_col'],
                se_wcs=src_info['wcs'],
                se_position_offset=src_info['position_offset'],
                start_row=start_row,
                start_col=start_col,
                coadd_wcs=coadd_info['wcs'],
                coadd_position_offset=coadd_info['position_offset']))

            psf, gs_psf = _build_psf_image(
                row=image_info['row'],
                col=image_info['col'],
                psf_rec=src_info['psf_rec'],
                galsim_wcs=src_info['galsim_wcs'],
                position_offset=src_info['position_offset'],
                coadding_interp=coadding_interp)
            gs_psfs.append(gs_psf)

            # build the coadding weight
            weights.append(_build_coadd_weight(
                coadding_weight=coadding_weight,
                weight=weight,
                psf=psf))

    if weights:
        weights = np.array(weights)
        weights /= np.sum(weights)

    return gs_images, bmasks, gs_noises, gs_psfs, weights, se_images_kept


def _coadd_slice_inputs(
        *, galsim_coadd_wcs, position_offset, row, col, start_row, start_col,
        box_size, psf_box_size, noise_interp_flags, se_interp_flags,
        images, bmasks, psfs, noises, weights):
    """Coadd the slice inputs to form the coadded image, noise realization,
    psf, and weight map.

    Note that the WCS of the output images is always the WCS at the
    input `(row, col)` for the input `galsim_coadd_wcs` using the
    `position_offset` to translate from the zero-indexed input `(row, col)`
    to the WCS indexing convention.

    Parameters
    ----------
    galsim_coadd_wcs
    position_offset : float
    row : float
    col : float
    start_row : int
    start_col : int
    box_size : int
    psf_box_size : int
    noise_interp_flags : int
    se_interp_flags : int
    images :
    noises :
    psfs :
    weights :

    Returns
    -------
    image : np.array
    bmask : np.array
    ormask : np.array
    noise : np.array
    psf : np.array
    weight : np.array
    """

    coadd_image = galsim.Image(
        box_size, box_size,
        dtype=np.float64,
        init_value=0,
        xmin=start_col + position_offset,
        ymin=start_row + position_offset,
        wcs=galsim_coadd_wcs)

    noise_image = galsim.Image(
        box_size, box_size,
        dtype=np.float64,
        init_value=0,
        xmin=start_col + position_offset,
        ymin=start_row + position_offset,
        wcs=galsim_coadd_wcs)

    _size = max(i.image.xmax - i.image.xmin + 1 for i in psfs)
    assert _size <= psf_box_size
    psf_start_col = col - ((psf_box_size - 1)/2)
    psf_start_row = row - ((psf_box_size - 1)/2)
    assert psf_start_col == int(psf_start_col)
    assert psf_start_row == int(psf_start_row)

    psf_image = galsim.Image(
        psf_box_size, psf_box_size,
        dtype=np.float64,
        init_value=0,
        xmin=int(psf_start_col) + position_offset,
        ymin=int(psf_start_row) + position_offset,
        wcs=galsim_coadd_wcs)

    img_sum = galsim.Sum([im * w for im, w in zip(images, weights)])
    img_sum.drawImage(image=coadd_image, method='no_pixel')

    img_sum = galsim.Sum([im * w for im, w in zip(noises, weights)])
    img_sum.drawImage(image=noise_image, method='no_pixel')

    img_sum = galsim.Sum([im * w for im, w in zip(psfs, weights)])
    img_sum.drawImage(image=psf_image, method='no_pixel')

    weight = np.zeros_like(noise_image.array)
    weight[:, :] = 1.0 / np.var(noise_image.array)

    ormask = np.zeros((box_size, box_size), dtype=np.int32)
    for coadd_bmask in bmasks:
        ormask |= coadd_bmask

    bmask = np.zeros((box_size, box_size), dtype=np.int32)
    for coadd_bmask in bmasks:
        msk = (coadd_bmask & noise_interp_flags) != 0
        bmask[msk] |= BMASK_NOISE_INTERP

        msk = (coadd_bmask & se_interp_flags) != 0
        bmask[msk] |= BMASK_SE_INTERP

    # make copies here to let python have a chance at freeing the galsim
    # objects
    return (
        coadd_image.array.copy(),
        bmask,
        ormask,
        noise_image.array.copy(),
        psf_image.array.copy(),
        weight)
