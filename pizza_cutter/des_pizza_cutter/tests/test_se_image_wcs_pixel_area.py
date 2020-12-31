import os

import numpy as np
import pytest

import galsim
import fitsio
import piff

from .._se_image import SEImageSlice, _get_wcs_area_interp, _compute_wcs_area

SE_DIMS_CUT = 512


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


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_get_wcs_pixel_area_pixmappy(se_image_data, coadd_image_data):
    se_wcs = piff.PSF.read(se_image_data['source_info']['piff_path']).wcs[0]

    # this hack mocks up an esutil-like interface to the pixmappy WCS
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

    def _is_celestial():
        return True

    se_wcs.image2sky = se_image2sky
    se_wcs.is_celestial = _is_celestial

    # we use the bit mask to exclude tape bumps and edges
    bmask = fitsio.read(
        se_image_data['source_info']['bmask_path'],
        ext=se_image_data['source_info']['bmask_ext'])
    bmask = bmask[:SE_DIMS_CUT, :SE_DIMS_CUT]

    # the apprixmate inversion assumes zero-indexed positions in and out
    _get_wcs_area_interp.cache_clear()
    wcs_inv = _get_wcs_area_interp(
        se_wcs,
        (SE_DIMS_CUT, SE_DIMS_CUT)
    )
    assert _get_wcs_area_interp.cache_info().hits == 0
    wcs_inv = _get_wcs_area_interp(
        se_wcs,
        (SE_DIMS_CUT, SE_DIMS_CUT)
    )
    assert _get_wcs_area_interp.cache_info().hits == 1

    rng = np.random.RandomState(seed=100)

    # all of the positions here are zero indexed
    x_se = rng.uniform(low=-0.5, high=bmask.shape[1]-0.5, size=1000)
    y_se = rng.uniform(low=-0.5, high=bmask.shape[0]-0.5, size=1000)
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

    pixel_area = _compute_wcs_area(
        se_wcs,
        x_se,
        y_se
    )

    pixel_area_intp = wcs_inv(x_se, y_se)

    err = np.sqrt((pixel_area_intp - pixel_area)**2)

    # everything should be finite
    assert np.all(np.isfinite(err)), np.sum(~np.isfinite(err))

    # in the interior the interp should be really good
    assert np.all(err[ok_pix] < 1e-4), np.max(err[ok_pix])

    # for the full image we allow more errors
    assert np.all(err < 5e-3), np.max(err)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_get_wcs_pixel_area_scamp(se_image_data, coadd_image_data):
    se_wcs = se_image_data['eu_wcs']

    # we use the bit mask to exclude tape bumps and edges
    bmask = fitsio.read(
        se_image_data['source_info']['bmask_path'],
        ext=se_image_data['source_info']['bmask_ext'])
    bmask = bmask[:SE_DIMS_CUT, :SE_DIMS_CUT]

    # the apprixmate inversion assumes zero-indexed positions in and out
    wcs_inv = _get_wcs_area_interp(
        se_wcs,
        (SE_DIMS_CUT, SE_DIMS_CUT),
        position_offset=1,
    )

    rng = np.random.RandomState(seed=100)

    # all of the positions here are zero indexed
    x_se = rng.uniform(low=-0.5, high=bmask.shape[1]-0.5, size=1000)
    y_se = rng.uniform(low=-0.5, high=bmask.shape[0]-0.5, size=1000)
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

    pixel_area = _compute_wcs_area(
        se_wcs,
        x_se+1,
        y_se+1,
    )

    pixel_area_intp = wcs_inv(x_se, y_se)

    err = np.sqrt((pixel_area_intp - pixel_area)**2)

    # everything should be finite
    assert np.all(np.isfinite(err)), np.sum(~np.isfinite(err))

    # in the interior the interp should be really good
    assert np.all(err[ok_pix] < 1e-4), np.max(err[ok_pix])

    # for the full image we allow more errors
    assert np.all(err < 5e-3), np.max(err)
