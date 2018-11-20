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


def _grid_interp(*, image, bad_msk):
    # this scale is a compromise between speed
    # - for images with very few pixels that require interpolation, smaller
    #    patches are better since we can skip more pixels
    # - for images that require a lot of interpolation, bigger patches are
    #   faster
    buff = 5
    size = 25

    # raise an error if the patches are not a clean multiple or the
    # image is not square
    n_patches = image.shape[0] / size
    assert n_patches == int(n_patches)
    n_patches = int(n_patches)
    assert image.shape[0] == image.shape[1]

    interp_image = image.copy()
    for i in range(n_patches):
        # total patch bounds and size
        ilow = max([i * size - buff, 0])
        ihigh = min([ilow + size + 2*buff, image.shape[0]])
        ni = ihigh - ilow

        # bounds of the final part ot be kept (no buffer) in the final image
        ilow_f = i * size
        ihigh_f = ilow_f + size

        # final part to be kept in the interpolated image
        ilow_s = buff if ilow_f != ilow else 0
        ihigh_s = size + ilow_s

        for j in range(n_patches):
            # total patch bounds and size
            jlow = max([j * size - buff, 0])
            jhigh = min([jlow + size + 2*buff, image.shape[0]])
            nj = jhigh - jlow

            # bounds of the final part ot be kept in the final image
            jlow_f = j * size
            jhigh_f = jlow_f + size

            # final part to be kept in the interpolated image
            jlow_s = buff if jlow_f != jlow else 0
            jhigh_s = size + jlow_s

            if np.any(bad_msk[ilow_f:ihigh_f, jlow_f:jhigh_f]):
                # only do interpolation if the final region needs it
                ii, jj = np.mgrid[0:ni, 0:nj]
                ii = ii.ravel()
                jj = jj.ravel()
                imr = image[ilow:ihigh, jlow:jhigh].ravel()
                bm = bad_msk[ilow:ihigh, jlow:jhigh].ravel()
                gm = ~bm
                yx = np.zeros((ii.size, 2))
                yx[:, 0] = ii
                yx[:, 1] = jj

                # now the real work begins
                _interp_imr = _interp_image(
                    image=imr,
                    good_msk=gm,
                    bad_msk=bm,
                    yx=yx).reshape((ni, nj))

                interp_image[ilow_f:ihigh_f, jlow_f:jhigh_f] \
                    = _interp_imr[ilow_s:ihigh_s, jlow_s:jhigh_s]
            elif np.all(bad_msk[ilow:ihigh, jlow:jhigh]):
                # if a whole patch is bad, then return None
                return None

    return interp_image


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

        interp_image = _grid_interp(image=image, bad_msk=bad_msk)
        if interp_image is None:
            return None, None

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
        return image.copy(), _draw_noise_image(weight=weight, rng=rng)
