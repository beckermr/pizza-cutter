import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

import numba
from numba import njit


def _draw_noise_image(*, weight, rng):
    """
    draw a noise image based on a weight map

    Parameters
    ----------
    weight: array
        A weight map, with no zeros
    rng: numpy.RandomState
        Random number generator
    """
    return rng.normal(size=weight.shape) * np.sqrt(1.0/weight)


@njit
def _get_nearby_good_pixels(image, bad_msk, nbad, buff=4):
    """
    get the set of good pixels surrounding bad pixels.

    Parameters
    ----------
    image: array
        The image data
    bad_msk: bool array
        2d array of mask bits.  True means it is a bad
        pixel

    Returns
    -------
    bad_pix:
        bad pix is the set of bad pixels, shape [nbad, 2]
    good_pix:
        good pix is the set of bood pixels around the bad
        pixels, shape [ngood, 2]
    good_im:
        the set of good image values, shape [ngood]
    good_ind:
        the 1d indices of the good pixels row*ncol + col
    """

    nrows, ncols = bad_msk.shape

    ngood = nbad*(2*buff+1)**2
    good_pix = np.zeros((ngood, 2), dtype=numba.int64)
    good_ind = np.zeros(ngood, dtype=numba.int64)
    bad_pix = np.zeros((ngood, 2), dtype=numba.int64)
    good_im = np.zeros(ngood, dtype=image.dtype)

    ibad = 0
    igood = 0
    for row in range(nrows):
        for col in range(ncols):
            val = bad_msk[row, col]
            if val:
                bad_pix[ibad] = (row, col)
                ibad += 1

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

                for rc in range(row_start, row_end+1):
                    for cc in range(col_start, col_end+1):
                        tval = bad_msk[rc, cc]
                        if not tval:

                            if igood == ngood:
                                raise RuntimeError('good_pix too small')

                            # got a good one, add it to the list
                            good_pix[igood] = (rc, cc)
                            good_im[igood] = image[rc, cc]

                            # keep track of index
                            ind = rc*ncols + cc
                            good_ind[igood] = ind
                            igood += 1

    bad_pix = bad_pix[:ibad, :]

    good_pix = good_pix[:igood, :]
    good_ind = good_ind[:igood]
    good_im = good_im[:igood]

    return bad_pix, good_pix, good_im, good_ind


def _grid_interp(*, image, bad_msk):
    """
    interpolate the bad pixels in an image

    Parameters
    ----------
    image: array
        the pixel data
    bad_msk: array
        boolean array, True means it is a bad pixel
    """

    nrows, ncols = image.shape
    npix = bad_msk.size

    nbad = bad_msk.sum()
    bm_frac = nbad/npix

    if bm_frac < 0.90:

        bad_pix, good_pix, good_im, good_ind = \
            _get_nearby_good_pixels(image, bad_msk, nbad)

        # extract unique ones
        gi, ind = np.unique(good_ind, return_index=True)

        good_pix = good_pix[ind, :]
        good_im = good_im[ind]

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


def interpolate_image_and_noise(
        *, image, weight, bmask, bad_flags, rng, noise=None):
    """Interpolate an image using the
    `scipy.interpolate.CloughTocher2DInterpolator`. An interpolated noise
    field is returned as well.

    Parameters
    ----------
    image : array-like
        The image to interpolate.
    weight : array-like
        The weight map of the image to interpolate.
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        Pixels with in the bit mask using
        `(bmask & bad_flags) != 0`.
    rng : `numpy.random.RandomState`
        An RNG instance to use.
    noise : array-like, optional
        Specify directly the noise field instead of using `rng` to generate
        one.

    Returns
    -------
    interp_image : array-like
        The interpolated image.
    interp_weight : array-like
        The interpolated weight map.
    """
    bad_msk = (weight <= 0) | ((bmask & bad_flags) != 0)

    if np.any(bad_msk):
        good_msk = ~bad_msk

        interp_image = _grid_interp(image=image, bad_msk=bad_msk)
        if interp_image is None:
            return None, None

        if noise is None:
            # fill the weight map with the median so we can draw a noise map
            # we could apply the interpolator too?
            interp_weight = weight.copy()
            if np.any(interp_weight[bad_msk] == 0):
                interp_weight[bad_msk] = np.median(interp_weight[good_msk])

            # now draw a noise map and apply an interp to it
            # this is to propagate how the interpolation correlates pixel noise
            # so it has to be done to the noise map
            noise = _draw_noise_image(weight=interp_weight, rng=rng)

        interp_noise = _grid_interp(image=noise, bad_msk=bad_msk)
        if interp_noise is None:
            return None, None

        return interp_image, interp_noise
    else:
        # return a copy here since the caller expects new images
        if noise is None:
            noise = _draw_noise_image(weight=weight, rng=rng)
        return image.copy(), noise
