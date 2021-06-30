import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import logging

from numba import njit

logger = logging.getLogger(__name__)


@njit
def _get_nearby_good_pixels(bad_msk, nbad, buff):
    """
    get the set of good pixels surrounding bad pixels.

    Parameters
    ----------
    bad_msk : bool array
        2d array of mask bits. True means it is a bad
        pixel
    nbad : int
        The number of bad pixels.
    buff : int
        The size of the good pixel buffer around each bad pixel.

    Returns
    -------
    bad_ind : array-like
        The 1d indices of the pixels to interp in row*ncol + col.
    bad_iso : array-like
        An array of 1 if the bad pixel doesn't have any buffer pixels which are ok, 0
        otherwise.
    good_ind : array-like
        The 1d indices of the good pixels to use in the interp in row*ncol + col.
    """

    nrows, ncols = bad_msk.shape

    ngood = nbad*(2*buff+1)**2
    bad_ind = np.zeros(ngood, dtype=np.int64)
    bad_iso = np.zeros(ngood, dtype=np.int64)
    good_ind = np.zeros(ngood, dtype=np.int64)
    no_good = 1

    ibad = 0
    igood = 0
    for row in range(nrows):
        for col in range(ncols):
            val = bad_msk[row, col]
            if val:
                bad_ind[ibad] = row * ncols + col

                row_start = row - buff
                row_end = row + buff
                col_start = col - buff
                col_end = col + buff

                if row_start < 0:
                    row_start = 0
                if row_end > (nrows-1):
                    row_end = nrows-1
                if col_start < 0:
                    col_start = 0
                if col_end > (ncols-1):
                    col_end = ncols-1

                no_good = 1
                for rc in range(row_start, row_end+1):
                    for cc in range(col_start, col_end+1):
                        tval = bad_msk[rc, cc]
                        if not tval:

                            if igood == ngood:
                                raise RuntimeError('good_pix too small')

                            ind = rc*ncols + cc
                            good_ind[igood] = ind
                            igood += 1
                            no_good = 0

                if no_good == 1:
                    bad_iso[ibad] = 1

                ibad += 1

    bad_ind = bad_ind[:ibad]
    bad_iso = bad_iso[:ibad]
    good_ind = good_ind[:igood]

    return bad_ind, bad_iso, good_ind


def interpolate_image_at_mask(*, image, bad_msk, maxfrac=0.90, buff=4):
    """
    interpolate the bad pixels in an image

    Parameters
    ----------
    image : array
        the pixel data
    bad_msk : array
        boolean array, True means it is a bad pixel
    maxfrac : float, optional
        If the fraction of bad pixels is greater than this,
        None is returned. Default is 0.90.
    buff : int, optional
        The buffer of good pixels around each bad pixel to keep for the interpolant.

    Returns
    -------
    interp_image : array-like
        The interpolated image.
    """
    nrows, ncols = image.shape
    npix = bad_msk.size

    nbad = bad_msk.sum()
    bm_frac = nbad/npix

    if bm_frac <= maxfrac:

        bad_ind, _, _good_ind = _get_nearby_good_pixels(bad_msk, nbad, buff)
        good_ind = np.unique(_good_ind)
        bad_yx = np.unravel_index(bad_ind, bad_msk.shape)
        good_yx = np.unravel_index(good_ind, bad_msk.shape)
        good_pix = np.array(good_yx).T
        bad_pix = np.array(bad_yx).T

        good_im = image[good_yx[0], good_yx[1]]

        img_interp = CloughTocher2DInterpolator(
            good_pix,
            good_im,
            fill_value=0.0,
        )
        interp_image = image.copy()
        interp_image[bad_msk] = img_interp(bad_pix)

        return interp_image

    else:
        return None


def _interp_one(
    *,
    image,
    bad_ind,
    bad_iso,
    good_ind,
    bad_yx,
    good_yx,
    bad_msk,
    noise_fill_yx,
    med_weight,
    rng,
):
    interp_image = image.copy()

    if noise_fill_yx is not None:
        shape = noise_fill_yx[0].shape
        interp_image[noise_fill_yx[0], noise_fill_yx[1]] = rng.normal(
            size=shape, scale=1.0/np.sqrt(med_weight)
        )

    img_interp = CloughTocher2DInterpolator(
        np.array(good_yx).T,
        interp_image[good_yx[0], good_yx[1]],
        fill_value=0.0,
    )
    interp_image[bad_msk] = img_interp(np.array(bad_yx).T)
    return interp_image


def interpolate_image_and_noise(
    *, image, noises, weight, bmask, bad_flags, rng,
    maxfrac=0.9, buff=4, fill_isolated_with_noise=False,
):
    """Interpolate an image using the
    `scipy.interpolate.CloughTocher2DInterpolator`. An interpolated noise
    field is returned as well.

    Parameters
    ----------
    image : array-like
        The image to interpolate.
    noises : list of array-like
        The noise fields to interpolate in the same way as the image.
    weight : array-like
        A weight map to test for zero values. Any pixels with zero weight
        are interpolated.
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        Pixels with in the bit mask using
        `(bmask & bad_flags) != 0`.
    rng : np.random.RandomState
        An RNG to use if we are filling isolkated bad pixels with noise.
    maxfrac : float, optional
        The maximum fraction of bad pixels. If the fraction is higher than this,
        None will be returned for the interpolated images.
    buff : int, optional
        The buffer of good pixels around each bad pixel to keep for the interpolant.
    fill_isolated_with_noise : bool, optional
        Fill isolated bad pixels with noise and then interp.

    Returns
    -------
    interp_image : array-like
        The interpolated image.
    interp_noises : list of array-like
        The interpolated noise field.
    """
    bad_msk = (weight <= 0) | ((bmask & bad_flags) != 0)

    if np.any(bad_msk):
        logger.debug('doing image interpolation')

        if np.mean(bad_msk) > maxfrac:
            return None, None

        med_weight = np.median(weight[~bad_msk])

        nbad = bad_msk.sum()
        bad_ind, bad_iso, _good_ind = _get_nearby_good_pixels(bad_msk, nbad, buff)
        good_ind = np.unique(_good_ind)
        bad_yx = np.unravel_index(bad_ind, bad_msk.shape)
        good_yx = np.unravel_index(good_ind, bad_msk.shape)

        if fill_isolated_with_noise:
            msk = bad_iso == 1
            if np.any(msk):
                # mark them as ok pixels
                bad_msk[bad_yx[0][msk], bad_yx[1][msk]] = False

                # keep the ones we have to fill
                noise_fill_yx = (bad_yx[0][msk], bad_yx[1][msk])

                # recompute the good pixels so that they inlcude the ones we
                # will noise fill
                nbad = bad_msk.sum()
                bad_ind, _, _good_ind = _get_nearby_good_pixels(bad_msk, nbad, buff)
                good_ind = np.unique(_good_ind)
                bad_yx = np.unravel_index(bad_ind, bad_msk.shape)
                good_yx = np.unravel_index(good_ind, bad_msk.shape)
        else:
            noise_fill_yx = None

        interp_image = _interp_one(
            image=image,
            bad_ind=bad_ind,
            bad_iso=bad_iso,
            good_ind=good_ind,
            bad_yx=bad_yx,
            good_yx=good_yx,
            bad_msk=bad_msk,
            noise_fill_yx=noise_fill_yx,
            rng=rng,
            med_weight=med_weight,
        )
        if interp_image is None:
            return None, None

        interp_noises = []
        for noise in noises:
            interp_noises.append(
                _interp_one(
                    image=noise,
                    bad_ind=bad_ind,
                    bad_iso=bad_iso,
                    good_ind=good_ind,
                    bad_yx=bad_yx,
                    good_yx=good_yx,
                    bad_msk=bad_msk,
                    noise_fill_yx=noise_fill_yx,
                    rng=rng,
                    med_weight=med_weight,
                )
            )
            if interp_noises[-1] is None:
                return None, None

        return interp_image, interp_noises
    else:
        # return a copy here since the caller expects new images
        return image.copy(), [noise.copy() for noise in noises]


def copy_masked_edges_image_and_noise(
    *, image, noises, weight, bmask, bad_flags
):
    """Any edge that is fully masked or has all weights zero
    has the adjacent pixels copied into it.

    Parameters
    ----------
    image : array-like
        The image to interpolate.
    noises : list of array-like
        The noise fields to interpolate in the same way as the image.
    weight : array-like
        A weight map to test for zero values. Any pixels with zero weight
        are interpolated.
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        Pixels with in the bit mask using
        `(bmask & bad_flags) != 0`.

    Returns
    -------
    interp_image : array-like
        The interpolated image.
    interp_noises : list of array-like
        The interpolated noise fields.
    interp_bmask : array-like
        The new bmask.
    interp_weight : array-like
        The new weight map.
    """
    interp_image = image.copy()
    interp_noises = [noise.copy() for noise in noises]
    interp_bmask, interp_weight = bmask.copy(), weight.copy()
    for i, ii in [(0, 1), (-1, -2)]:
        bad_msk_i = (weight[i, :] <= 0) | ((bmask[i, :] & bad_flags) != 0)
        bad_msk_ii = (weight[ii, :] <= 0) | ((bmask[ii, :] & bad_flags) != 0)
        if np.all(bad_msk_i) and not np.all(bad_msk_ii):
            logger.debug('doing msk edge interpolation row %d', i)

            interp_image[i, ~bad_msk_ii] = interp_image[ii, ~bad_msk_ii]
            interp_bmask[i, ~bad_msk_ii] = interp_bmask[ii, ~bad_msk_ii]
            interp_weight[i, ~bad_msk_ii] = interp_weight[ii, ~bad_msk_ii]
            for k in range(len(noises)):
                interp_noises[k][i, ~bad_msk_ii] = interp_noises[k][ii, ~bad_msk_ii]

        bad_msk_i = (weight[:, i] <= 0) | ((bmask[:, i] & bad_flags) != 0)
        bad_msk_ii = (weight[:, ii] <= 0) | ((bmask[:, ii] & bad_flags) != 0)
        if np.all(bad_msk_i) and not np.all(bad_msk_ii):
            logger.debug('doing msk edge interpolation col %d', i)
            interp_image[~bad_msk_ii, i] = interp_image[~bad_msk_ii, ii]
            interp_bmask[~bad_msk_ii, i] = interp_bmask[~bad_msk_ii, ii]
            interp_weight[~bad_msk_ii, i] = interp_weight[~bad_msk_ii, ii]
            for k in range(len(noises)):
                interp_noises[k][~bad_msk_ii, i] = interp_noises[k][~bad_msk_ii, ii]

    return interp_image, interp_noises, interp_bmask, interp_weight
