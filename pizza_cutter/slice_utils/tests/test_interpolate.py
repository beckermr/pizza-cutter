import pytest
import numpy as np
from ..interpolate import interpolate_image_and_noise


@pytest.mark.parametrize('use_rng', [True, False])
def test_interpolate_image_and_noise_weight(use_rng):
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 0
    rng = np.random.RandomState(seed=42)
    weight[30:35, 40:45] = 0.0

    # put nans here to make sure interp is done ok
    msk = weight <= 0
    image[msk] = np.nan

    if use_rng:
        iimage, inoise = interpolate_image_and_noise(
            image=image,
            weight=weight,
            bmask=bmask,
            bad_flags=bad_flags,
            rng=rng)
    else:
        rng = np.random.RandomState(seed=42)
        noise = rng.normal(size=image.shape)
        iimage, inoise = interpolate_image_and_noise(
            image=image,
            weight=weight,
            bmask=bmask,
            bad_flags=bad_flags,
            rng=None,  # this will cause an error if the RNG is called
            noise=noise)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])


@pytest.mark.parametrize('use_rng', [True, False])
def test_interpolate_image_and_noise_bmask(use_rng):
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 1

    rng = np.random.RandomState(seed=42)
    bmask[30:35, 40:45] = 1
    bmask[:, 0] = 2
    bmask[:, -1] = 4

    # put nans here to make sure interp is done ok
    msk = (bmask & bad_flags) != 0
    image[msk] = np.nan

    if use_rng:
        iimage, inoise = interpolate_image_and_noise(
            image=image,
            weight=weight,
            bmask=bmask,
            bad_flags=bad_flags,
            rng=rng)
    else:
        rng = np.random.RandomState(seed=42)
        noise = rng.normal(size=image.shape)
        iimage, inoise = interpolate_image_and_noise(
            image=image,
            weight=weight,
            bmask=bmask,
            bad_flags=bad_flags,
            rng=None,  # this will cause an error if the RNG is called
            noise=noise)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_image_and_noise_big_missing():
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 1

    rng = np.random.RandomState(seed=42)
    bmask[15:80, 15:80] = 1

    # put nans here to make sure interp is done ok
    msk = (bmask & bad_flags) != 0
    image[msk] = np.nan

    iimage, inoise = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        rng=rng)

    # interp will be waaay off but shpuld have happened
    assert np.all(np.isfinite(iimage))

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_gauss_image(show=False):
    """
    test that our interpolation works decently for a linear
    piece missing from a gaussian image
    """

    rng = np.random.RandomState(seed=31415)
    noise = 0.001

    sigma = 4.0
    is2 = 1.0/sigma**2
    dims = 51, 51
    cen = (np.array(dims)-1.0)/2.0

    rows, cols = np.mgrid[
        0:dims[0],
        0:dims[1],
    ]
    rows = rows - cen[0]
    cols = cols - cen[1]

    image_unmasked = np.exp(-0.5*(rows**2 + cols**2)*is2)
    weight = image_unmasked*0 + 1.0/noise**2

    noise_image = rng.normal(scale=noise, size=image_unmasked.shape)

    badcol = int(cen[1]-3)
    bw = 3
    rr = badcol-bw, badcol+bw+1

    weight[rr[0]:rr[1], badcol] = 0.0
    image_masked = image_unmasked.copy()
    image_masked[rr[0]:rr[1], badcol] = 0.0

    bmask = np.zeros_like(image_unmasked, dtype=np.int32)
    bad_flags = 0

    iimage, inoise = interpolate_image_and_noise(
        image=image_masked,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        rng=None,  # this will cause an error if the RNG is called
        noise=noise_image,
    )

    maxdiff = np.abs(image_unmasked-iimage).max()

    if show:
        import images
        images.view_mosaic([image_masked, weight])

        images.compare_images(
            image_unmasked,
            iimage,
            width=2000,
            height=int(2000*2/3),
        )
        print('max diff:', maxdiff)

    assert maxdiff < 0.0015
