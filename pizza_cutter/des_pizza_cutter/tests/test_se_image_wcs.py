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
def test_se_image_wcs_array_shape(se_image_data, x, y):
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
        se_im.image2sky(x, y)

    with pytest.raises(AssertionError):
        se_im.image2sky(y, x)

    with pytest.raises(AssertionError):
        se_im.sky2image(x, y)

    with pytest.raises(AssertionError):
        se_im.sky2image(y, x)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_esutil(se_image_data, wcs_pos_offset):
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
        ra, dec = se_im.image2sky(x, y)
        eu_ra, eu_dec = se_image_data['eu_wcs'].image2sky(
            x+wcs_pos_offset, y+wcs_pos_offset)
        assert np.allclose(ra, eu_ra)
        assert np.allclose(dec, eu_dec)

        x, y = se_im.sky2image(ra, dec)
        eu_x, eu_y = se_image_data['eu_wcs'].sky2image(ra, dec)
        assert np.allclose(x, eu_x - wcs_pos_offset)
        assert np.allclose(y, eu_y - wcs_pos_offset)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_esutil_array(se_image_data, wcs_pos_offset):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        noise_seeds=[10],
        mask_tape_bumps=False,
    )

    x = np.arange(10)
    y = np.arange(10)
    ra, dec = se_im.image2sky(x, y)
    assert ra.shape == x.shape
    assert dec.shape == x.shape

    eu_ra, eu_dec = se_image_data['eu_wcs'].image2sky(
        x+wcs_pos_offset, y+wcs_pos_offset)
    assert np.allclose(ra, eu_ra)
    assert np.allclose(dec, eu_dec)

    x, y = se_im.sky2image(ra, dec)
    assert x.shape == ra.shape
    assert y.shape == ra.shape

    eu_x, eu_y = se_image_data['eu_wcs'].sky2image(ra, dec)
    assert np.allclose(x, eu_x - wcs_pos_offset)
    assert np.allclose(y, eu_y - wcs_pos_offset)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_galsim(se_image_data, wcs_pos_offset):
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
        ra, dec = se_im.image2sky(x, y)
        pos = se_image_data['gs_wcs'].toWorld(
            galsim.PositionD(x=x+wcs_pos_offset, y=y+wcs_pos_offset))
        assert np.allclose(ra, pos.ra / galsim.degrees)
        assert np.allclose(dec, pos.dec / galsim.degrees)

        x, y = se_im.sky2image(ra, dec)
        wpos = galsim.CelestialCoord(
            ra=ra * galsim.degrees,
            dec=dec * galsim.degrees)
        pos = se_image_data['gs_wcs'].toImage(wpos)
        assert np.allclose(x, pos.x - wcs_pos_offset)
        assert np.allclose(y, pos.y - wcs_pos_offset)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_galsim_array(se_image_data, wcs_pos_offset):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['gs_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        noise_seeds=[10],
        mask_tape_bumps=False,
    )

    x = np.arange(2)
    y = np.arange(2) + 100

    ra, dec = se_im.image2sky(x, y)
    x_out, y_out = se_im.sky2image(ra, dec)

    for i in range(2):
        pos = se_image_data['gs_wcs'].toWorld(
            galsim.PositionD(x=x[i]+wcs_pos_offset, y=y[i]+wcs_pos_offset))
        assert np.allclose(ra[i], pos.ra / galsim.degrees)
        assert np.allclose(dec[i], pos.dec / galsim.degrees)

        wpos = galsim.CelestialCoord(
            ra=ra[i] * galsim.degrees,
            dec=dec[i] * galsim.degrees)
        pos = se_image_data['gs_wcs'].toImage(wpos)
        assert np.allclose(x[i], pos.x - wcs_pos_offset)
        assert np.allclose(y[i], pos.y - wcs_pos_offset)
