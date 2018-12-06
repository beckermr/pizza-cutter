import copy
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


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


def _interp_patch(*, image, bad_msk, i, j, size, buff):
    # total patch bounds and size
    ilow = max([i * size - buff, 0])
    ihigh = min([ilow + size + 2*buff, image.shape[0]])
    ni = ihigh - ilow

    # bounds of the final part ot be kept (no buffer) in the final image
    ilow_f = i * size
    # ihigh_f = ilow_f + size  # not needed

    # final part to be kept in the interpolated image
    ilow_s = buff if ilow_f != ilow else 0
    ihigh_s = size + ilow_s

    # total patch bounds and size
    jlow = max([j * size - buff, 0])
    jhigh = min([jlow + size + 2*buff, image.shape[0]])
    nj = jhigh - jlow

    # bounds of the final part ot be kept in the final image
    jlow_f = j * size
    # jhigh_f = jlow_f + size  # not needed

    # final part to be kept in the interpolated image
    jlow_s = buff if jlow_f != jlow else 0
    jhigh_s = size + jlow_s

    npix = bad_msk[ilow:ihigh, jlow:jhigh].size
    bm_sum = np.sum(bad_msk[ilow:ihigh, jlow:jhigh])
    bm_frac = bm_sum / npix

    if bm_frac < 0.90 and npix - bm_sum > 10:
        # only do interpolation if the final region needs it and we have
        # enough data
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

        return _interp_imr[ilow_s:ihigh_s, jlow_s:jhigh_s]
    else:
        # signals to caller that we need to try again with a bigger buffer
        return None


def _grid_interp(*, image, bad_msk):
    # this scale is a compromise between speed
    # - for images with very few pixels that require interpolation, smaller
    #    patches are better since we can skip more pixels
    # - for images that require a lot of interpolation, bigger patches are
    #   faster
    buff_start = 5
    delta_buff = 5
    size = 25

    # raise an error if the patches are not a clean multiple or the
    # image is not square
    n_patches = image.shape[0] / size
    assert n_patches == int(n_patches)
    n_patches = int(n_patches)
    assert image.shape[0] == image.shape[1]

    interp_image = image.copy()
    for i in range(n_patches):
        # bounds of the final part ot be kept (no buffer) in the final image
        ilow_f = i * size
        ihigh_f = ilow_f + size

        for j in range(n_patches):
            # bounds of the final part ot be kept in the final image
            jlow_f = j * size
            jhigh_f = jlow_f + size

            if np.any(bad_msk[ilow_f:ihigh_f, jlow_f:jhigh_f]):
                # set a flag here - if we fail the interp, then we raise
                did_interp = False

                buff = copy.copy(buff_start)
                while buff <= size * 2:
                    imint = _interp_patch(
                        image=image,
                        bad_msk=bad_msk,
                        i=i,
                        j=j,
                        size=size,
                        buff=buff)

                    if imint is None:
                        # too few pixels, increase the buffer
                        buff += delta_buff
                    else:
                        # yay - set flag, image and break out
                        did_interp = True
                        interp_image[ilow_f:ihigh_f, jlow_f:jhigh_f] = imint
                        break

                if not did_interp:
                    raise RuntimeError(
                        "Interpolation failed: buff|i|j = %d|%d|%d" % (
                            buff, i, j))

    return interp_image


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
