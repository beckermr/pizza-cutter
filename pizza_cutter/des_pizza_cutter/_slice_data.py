import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


def build_slice_locations(*, central_size, buffer_size, image_width):
    """Build the locations of the slices.

    Parameters
    ----------
    central_size : int
        Size of the central region for metadetection in pixels.
    buffer_size : int
        Size of the buffer around the central region in pixels.
    image_width : int
        The width of the image in pixels.

    Returns
    -------
    row : array-like
        The row of the slice centers.
    col : array-like
        The column of the slice centers.
    start_row : array-like
        The starting row of the slice.
    start_col : array-like
        The starting column of the slice.
    """
    # compute the number of pizza slices
    # - each slice is central_size + 2 * buffer_size wide
    #   because a buffer is on each side
    # - each pizza slice is moved over by central_size to tile the full
    #  coadd tile
    # - we leave a buffer on either side so that we have to tile only
    #  im_width - 2 * buffer_size pixels
    cen_offset = (central_size - 1) / 2.0  # zero-indexed at pixel centers

    nobj = (image_width - 2 * buffer_size) / central_size
    if int(nobj) != nobj:
        raise ValueError(
            "The coadd tile must be exactly tiled by the pizza slices!")
    nobj = int(nobj)

    loc = np.arange(nobj) * central_size + buffer_size + cen_offset
    col, row = np.meshgrid(loc, loc)
    row = row.ravel()
    col = col.ravel()
    start_row = row - buffer_size - cen_offset
    start_col = col - buffer_size - cen_offset

    return row, col, start_row, start_col


def symmetrize_weight(*, weight):
    """Symmetrize zero weight pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    weight : array-like
        The weight map for the slice.
    """
    if weight.shape[0] != weight.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    weight_rot = np.rot90(weight)
    msk = weight_rot == 0.0
    if np.any(msk):
        weight[msk] = 0.0


def symmetrize_bmask(*, bmask, bad_flags):
    """Symmetrize masked pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        The flags to symmetrize in the bit mask using
        `(bmask & bad_flags) != 0`.
    """
    if bmask.shape[0] != bmask.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    bm_rot = np.rot90(bmask)
    msk = (bm_rot & bad_flags) != 0
    if np.any(msk):
        bmask[msk] |= bm_rot[msk]


def _interp_image(*, image, good_msk, bad_msk, yx):
    img_interp = CloughTocher2DInterpolator(
        yx[good_msk, :],
        image[good_msk],
        fill_value=0.0)
    interp_image = image.copy()
    interp_image[bad_msk] = img_interp(yx[bad_msk, :])
    return interp_image


def _draw_noise_image(*, weight, rng):
    return rng.normal(size=weight.shape) * np.sqrt(1.0/weight)


def interpolate_image_and_noise(*, image, weight, bmask, bad_flags, rng):
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

        # ravel it all here
        # eveything past here is done on the raveled image
        good_msk = good_msk.ravel()
        bad_msk = bad_msk.ravel()

        # we need the coordinates for the image interpolator
        y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        x = x.ravel()
        y = y.ravel()
        yx = np.zeros((x.size, 2))
        yx[:, 0] = y
        yx[:, 1] = x

        # now the real work begins
        interp_image = _interp_image(
            image=image.ravel(),
            good_msk=good_msk,
            bad_msk=bad_msk,
            yx=yx)

        # fill the weight map with the median so we can draw a noise map
        # we could apply the interpolator too?
        interp_weight = weight.copy().ravel()
        if np.any(interp_weight[bad_msk] == 0):
            interp_weight[bad_msk] = np.median(interp_weight[good_msk])

        # now draw a noise map and apply an interp to it
        # this is to propagate how the interpolation correlates pixel noise
        # so it has to be done to the noise map
        noise = _draw_noise_image(weight=interp_weight, rng=rng)
        interp_noise = _interp_image(
            image=noise,
            good_msk=good_msk,
            bad_msk=bad_msk,
            yx=yx)

        # now unravel
        interp_image = interp_image.reshape(image.shape)
        interp_noise = interp_noise.reshape(image.shape)

        return interp_image, interp_noise
    else:
        # return a copy here since the caller expects new images
        return image.copy(), _draw_noise_image(weight=weight, rng=rng)
