import os
import numpy as np
import pytest
import fitsio

import piff

from .._se_image import _get_wcs_inverse


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_get_wcs_inverse_pixmappy(se_image_data, coadd_image_data):
    coadd_wcs = coadd_image_data['eu_wcs']

    se_wcs = piff.PSF.read(se_image_data['source_info']['piff_path']).wcs[0]

    # this hack mocks up an esutil=like interface to the pixmappy WCS
    def se_image2sky(x, y):
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            is_scalar = True
        else:
            is_scalar = False
        # the factor of +1 here converts from zero to one indexed
        ra, dec = se_wcs._radec(
            (np.atleast_1d(x) - se_wcs.x0 +
             se_image_data['source_info']['position_offset']),
            (np.atleast_1d(y) - se_wcs.y0 +
             se_image_data['source_info']['position_offset']))
        np.degrees(ra, out=ra)
        np.degrees(dec, out=dec)
        if is_scalar:
            return ra[0], dec[0]
        else:
            return ra, dec

    se_wcs.image2sky = se_image2sky

    # we use the bit mask to exclude tape bumps and edges
    bmask = fitsio.read(
        se_image_data['source_info']['bmask_path'],
        ext=se_image_data['source_info']['bmask_ext'])

    # the apprixmate inversion assumes zero-indexed positions in and out
    wcs_inv = _get_wcs_inverse(
        coadd_wcs,
        coadd_image_data['position_offset'],
        se_wcs,
        se_image_data['source_info']['position_offset'],
        (4096, 2048)
    )

    rng = np.random.RandomState(seed=100)

    # all of the positions here are zero indexed
    x_se = rng.uniform(low=-0.5, high=bmask.shape[1]-0.5, size=100000)
    y_se = rng.uniform(low=-0.5, high=bmask.shape[0]-0.5, size=100000)
    x_se_pix = (x_se + 0.5).astype(np.int64)
    y_se_pix = (y_se + 0.5).astype(np.int64)
    bmask_vals = bmask[y_se_pix, x_se_pix]

    # we demand good interpolations where there are
    # 1) no suspect pixels (eg tape bumps for Y3, bit 2048)
    # 2) no edges (bit 512)
    # 3) not in the 64 pixels around the edge of the CCD
    buff = 64
    ok_pix = (
        ((bmask_vals & 2048) == 0) &
        ((bmask_vals & 512) == 0) &
        (x_se_pix >= buff) &
        (x_se_pix < bmask.shape[1] - buff) &
        (y_se_pix >= buff) &
        (y_se_pix < bmask.shape[0] - buff))

    coadd_pos = coadd_wcs.sky2image(*se_wcs.image2sky(x_se, y_se))
    # outputs are one-indexed so we convert back
    coadd_x = coadd_pos[0] - coadd_image_data['position_offset']
    coadd_y = coadd_pos[1] - coadd_image_data['position_offset']

    x_se_intp, y_se_intp = wcs_inv(coadd_x, coadd_y)

    err = np.sqrt(
        ((x_se - x_se_intp) * 0.263)**2 +
        ((y_se - y_se_intp) * 0.263)**2)

    # everything should be finite
    assert np.all(np.isfinite(err)), np.sum(~np.isfinite(err))

    # in the interior the interp should be really good
    assert np.all(err[ok_pix] < 1e-3), np.max(err[ok_pix])

    # for the full image we allow more errors
    assert np.all(err < 2e-2), np.max(err)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_get_wcs_inverse_scamp(se_image_data, coadd_image_data):
    coadd_wcs = coadd_image_data['eu_wcs']

    se_wcs = se_image_data['eu_wcs']

    # we use the bit mask to exclude tape bumps and edges
    bmask = fitsio.read(
        se_image_data['source_info']['bmask_path'],
        ext=se_image_data['source_info']['bmask_ext'])

    # the apprixmate inversion assumes zero-indexed positions in and out
    wcs_inv = _get_wcs_inverse(
        coadd_wcs,
        coadd_image_data['position_offset'],
        se_wcs,
        se_image_data['source_info']['position_offset'],
        (4096, 2048)
    )

    rng = np.random.RandomState(seed=100)

    # all of the positions here are zero indexed
    x_se = rng.uniform(low=-0.5, high=bmask.shape[1]-0.5, size=100000)
    y_se = rng.uniform(low=-0.5, high=bmask.shape[0]-0.5, size=100000)
    x_se_pix = (x_se + 0.5).astype(np.int64)
    y_se_pix = (y_se + 0.5).astype(np.int64)
    bmask_vals = bmask[y_se_pix, x_se_pix]

    # we demand good interpolations where there are
    # 1) no suspect pixels (eg tape bumps for Y3, bit 2048)
    # 2) no edges (bit 512)
    # 3) not in the 64 pixels around the edge of the CCD
    buff = 64
    ok_pix = (
        ((bmask_vals & 2048) == 0) &
        ((bmask_vals & 512) == 0) &
        (x_se_pix >= buff) &
        (x_se_pix < bmask.shape[1] - buff) &
        (y_se_pix >= buff) &
        (y_se_pix < bmask.shape[0] - buff))

    coadd_pos = coadd_wcs.sky2image(*se_wcs.image2sky(
        x_se+se_image_data['source_info']['position_offset'],
        y_se+se_image_data['source_info']['position_offset']))
    # outputs are one-indexed so we convert back
    coadd_x = coadd_pos[0] - coadd_image_data['position_offset']
    coadd_y = coadd_pos[1] - coadd_image_data['position_offset']

    x_se_intp, y_se_intp = wcs_inv(coadd_x, coadd_y)

    err = np.sqrt(
        ((x_se - x_se_intp) * 0.263)**2 +
        ((y_se - y_se_intp) * 0.263)**2)

    # everything should be finite
    assert np.all(np.isfinite(err)), np.sum(~np.isfinite(err))

    # in the interior the interp should be really good
    assert np.all(err[ok_pix] < 1e-3), np.max(err[ok_pix])

    # for the full image we allow more errors
    assert np.all(err < 2e-2), np.max(err)
