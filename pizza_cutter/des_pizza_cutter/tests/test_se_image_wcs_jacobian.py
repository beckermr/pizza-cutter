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
def test_se_image_wcs_jacobian_array(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        noise_seeds=[10],
        mask_tape_bumps=False,
    )

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(x, y)

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(y, x)

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(x, y)

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(y, x)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_jacobian_esutil(se_image_data, wcs_pos_offset):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        noise_seeds=[10],
        mask_tape_bumps=False,
    )

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        jac = se_im.get_wcs_jacobian(x, y)
        tup = se_image_data['eu_wcs'].get_jacobian(
            x+wcs_pos_offset, y+wcs_pos_offset)
        assert jac.dudx == tup[0]
        assert jac.dudy == tup[1]
        assert jac.dvdx == tup[2]
        assert jac.dvdy == tup[3]


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_jacobian_galsim(se_image_data, wcs_pos_offset):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['gs_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        noise_seeds=[10],
        mask_tape_bumps=False,
    )

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        jac = se_im.get_wcs_jacobian(x, y)
        gs_jac = se_image_data['gs_wcs'].local(
            galsim.PositionD(x=x+wcs_pos_offset, y=y+wcs_pos_offset))
        assert jac.dudx == gs_jac.dudx
        assert jac.dudy == gs_jac.dudy
        assert jac.dvdx == gs_jac.dvdx
        assert jac.dvdy == gs_jac.dvdy
