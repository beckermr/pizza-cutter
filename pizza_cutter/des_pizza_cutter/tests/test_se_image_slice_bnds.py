import os

import pytest

from .._se_image import SEImageSlice


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('eps', [-0.5, -0.1, 0, 0.1, 0.499999])
def test_se_image_slice_bnds_odd(se_image_data, eps):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    row_cen = 16
    col_cen = 19
    box_size = 21
    box_cen = (box_size - 1) / 2
    ra, dec = se_im.image2sky(col_cen+eps, row_cen+eps)
    bnds = se_im.compute_slice_bounds(ra, dec, box_size)

    assert bnds.rowmin == row_cen - box_cen
    assert bnds.rowmax == row_cen + box_cen
    assert bnds.colmin == col_cen - box_cen
    assert bnds.colmax == col_cen + box_cen


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('eps', [0.0000001, 0.1, 0.5, 0.7, 0.99999])
def test_se_image_slice_bnds_even(se_image_data, eps):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    row_cen = 16
    col_cen = 19
    box_size = 20
    box_cen = (box_size - 1) / 2
    ra, dec = se_im.image2sky(col_cen+eps, row_cen+eps)
    bnds = se_im.compute_slice_bounds(ra, dec, box_size)

    assert bnds.rowmin == row_cen - box_cen + 0.5
    assert bnds.rowmax == row_cen + box_cen + 0.5
    assert bnds.colmin == col_cen - box_cen + 0.5
    assert bnds.colmax == col_cen + box_cen + 0.5
