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
