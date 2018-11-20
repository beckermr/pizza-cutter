import numpy as np
import pytest

from .._slice_data import (
    symmetrize_bmask,
    symmetrize_weight,
    build_slice_locations,
    interpolate_image_and_noise)


def test_build_slice_locations():
    row, col, start_row, start_col = build_slice_locations(
        central_size=20, buffer_size=10, image_width=100)

    for i in range(4):
        for j in range(4):
            ind = j + 4*i
            assert row[ind] == i*20 + 10 + 9.5
            assert col[ind] == j*20 + 10 + 9.5
            assert start_row[ind] == i * 20
            assert start_col[ind] == j * 20


def test_build_slice_locations_error():
    with pytest.raises(ValueError):
        build_slice_locations(
            central_size=15, buffer_size=10, image_width=100)


def test_symmetrize_weight():
    weight = np.ones((5, 5))
    weight[:, 0] = 0
    symmetrize_weight(weight=weight)

    assert np.all(weight[:, 0] == 0)
    assert np.all(weight[-1, :] == 0)


def test_symmetrize_bmask():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bad_flags = 1
    bmask[:, 0] = bad_flags
    bmask[:, -2] = 2
    symmetrize_bmask(bmask=bmask, bad_flags=bad_flags)

    assert np.array_equal(
        bmask,
        [[1, 0, 2, 0],
         [1, 0, 2, 0],
         [1, 0, 2, 0],
         [1, 1, 3, 1]])


def test_interpolate_image_and_noise_weight():
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

    iimage, inoise = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        rng=rng)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_image_and_noise_bmask():
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

    iimage, inoise = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        rng=rng)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoise[msk])
    assert np.allclose(noise[~msk], inoise[~msk])
