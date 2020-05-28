import os

import numpy as np
import pytest

import galsim

from .._se_image import SEImageSlice


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('x,y', [
    (np.ones(10), 10),
    (np.ones((10, 10)), 10)])
def test_se_image_wcs_pixel_area_array(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    with pytest.raises(AssertionError):
        se_im.get_wcs_pixel_area(x, y)

    with pytest.raises(AssertionError):
        se_im.get_wcs_pixel_area(y, x)

    with pytest.raises(AssertionError):
        se_im.get_wcs_pixel_area(x, y)

    with pytest.raises(AssertionError):
        se_im.get_wcs_pixel_area(y, x)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_pixel_area_esutil(se_image_data, wcs_pos_offset):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        area = se_im.get_wcs_pixel_area(x, y)
        tup = se_image_data['eu_wcs'].get_jacobian(
            x+wcs_pos_offset, y+wcs_pos_offset)
        assert np.allclose(area, np.abs(tup[2] * tup[1] - tup[0] * tup[3]))

    xa = rng.uniform(size=10) * 2048
    ya = rng.uniform(size=10) * 4096
    area = se_im.get_wcs_pixel_area(xa, ya)
    for i, (x, y) in enumerate(zip(xa, ya)):
        tup = se_image_data['eu_wcs'].get_jacobian(
            x+wcs_pos_offset, y+wcs_pos_offset)
        assert np.allclose(area[i], np.abs(tup[2] * tup[1] - tup[0] * tup[3]))


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_pixel_area_galsim(se_image_data, wcs_pos_offset):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['gs_wcs'],
        wcs_position_offset=wcs_pos_offset,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        area = se_im.get_wcs_pixel_area(x, y)
        gs_area = se_image_data['gs_wcs'].local(
            galsim.PositionD(x=x+wcs_pos_offset, y=y+wcs_pos_offset)).pixelArea()
        assert np.allclose(area, gs_area)

    xa = rng.uniform(size=10) * 2048
    ya = rng.uniform(size=10) * 4096
    area = se_im.get_wcs_pixel_area(xa, ya)
    for i, (x, y) in enumerate(zip(xa, ya)):
        gs_area = se_image_data['gs_wcs'].local(
            galsim.PositionD(x=x+wcs_pos_offset, y=y+wcs_pos_offset)).pixelArea()
        assert np.allclose(area[i], gs_area)
