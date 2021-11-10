import os

import numpy as np
import pytest

from .._se_image import SEImageSlice
from .._affine_wcs import AffineWCS


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('x,y', [
    (np.ones(10), 10),
    (np.ones((10, 10)), 10)])
def test_se_image_sky_bnds_array_shape(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    with pytest.raises(AssertionError):
        se_im.image2sky(x, y)

    with pytest.raises(AssertionError):
        se_im.ccd_contains_radec(y, x)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_sky_bnds_array(se_image_data):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    ra = se_im._ra_ccd * np.ones(10)
    dec = se_im._dec_ccd * np.ones(10)
    dec[0] += 15.0  # not in the CCD

    msk = se_im.ccd_contains_radec(ra, dec)
    assert not msk[0]
    assert np.all(msk[1:])


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_sky_bnds_scalar(se_image_data):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    ra = se_im._ra_ccd
    dec = se_im._dec_ccd
    msk = se_im.ccd_contains_radec(ra, dec)
    assert msk
    assert msk is True

    dec += 15.0  # not in the CCD
    msk = se_im.ccd_contains_radec(ra, dec)
    assert not msk
    assert msk is False


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_sky_bnds_affine_array(se_image_data):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=AffineWCS(
            dudx=0.263, dudy=-0.002, dvdx=0.002, dvdy=0.263, x0=3, y0=1),
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    ra = se_im._ra_ccd * np.ones(10)
    dec = se_im._dec_ccd * np.ones(10)
    dec[0] += 1e9  # not in the CCD

    msk = se_im.ccd_contains_radec(ra, dec)
    assert not msk[0]
    assert np.all(msk[1:])


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_sky_bnds_affine_scalar(se_image_data):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=AffineWCS(
            dudx=0.263, dudy=-0.002, dvdx=0.002, dvdy=0.263, x0=3, y0=1),
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    ra = se_im._ra_ccd
    dec = se_im._dec_ccd
    msk = se_im.ccd_contains_radec(ra, dec)
    assert msk
    assert msk is True

    dec += 1e9  # not in the CCD
    msk = se_im.ccd_contains_radec(ra, dec)
    assert not msk
    assert msk is False
