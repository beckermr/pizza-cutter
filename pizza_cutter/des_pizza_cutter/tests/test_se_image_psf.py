import os
import numpy as np
import pytest

import galsim
import piff

from .._se_image import SEImageSlice, PIFF_STAMP_SIZE


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('x,y', [
    (np.ones(10), 10),
    (np.ones((10, 10)), 10)])
def test_se_image_psf_array(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    with pytest.raises(AssertionError):
        se_im.get_psf_image(x, y)

    with pytest.raises(AssertionError):
        se_im.get_psf_image(y, x)

    with pytest.raises(AssertionError):
        se_im.get_psf_image(x, y)

    with pytest.raises(AssertionError):
        se_im.get_psf_image(y, x)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
@pytest.mark.parametrize('eps_x', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize('eps_y', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
def test_se_image_psf_gsobject(se_image_data, eps_x, eps_y, wcs_pos_offset):
    x = 10 + eps_x
    y = 11 + eps_y
    dx = x - int(x + 0.5)
    dy = y - int(y + 0.5)

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=galsim.Gaussian(fwhm=0.8),
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    psf_im = se_im.get_psf_image(x, y)
    cen = (psf_im.shape[0] - 1) / 2

    # check mean (x, y) to make sure it is in the right spot
    _y, _x = np.mgrid[:psf_im.shape[0], :psf_im.shape[1]]
    xbar = np.mean((_x - cen) * psf_im) / np.mean(psf_im)
    ybar = np.mean((_y - cen) * psf_im) / np.mean(psf_im)
    assert np.abs(xbar - dx) < 1e-3, xbar
    assert np.abs(ybar - dy) < 1e-3, ybar

    true_psf_im = galsim.Gaussian(fwhm=0.8).drawImage(
        nx=19,
        ny=19,
        wcs=se_im.get_wcs_jacobian(x, y),
        offset=galsim.PositionD(x=dx, y=dy)
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
@pytest.mark.parametrize('eps_x', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize('eps_y', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize('use_wcs', [False, True])
def test_se_image_psf_psfex(
        se_image_data, use_wcs, eps_x, eps_y, wcs_pos_offset):
    if use_wcs:
        psfex = galsim.des.DES_PSFEx(
            se_image_data['source_info']['psf_path'],
            se_image_data['source_info']['image_path'],
            )
    else:
        psfex = galsim.des.DES_PSFEx(
            se_image_data['source_info']['psf_path'])

    x = 10 + eps_x
    y = 11 + eps_y
    dx = x - int(x + 0.5)
    dy = y - int(y + 0.5)

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psfex,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    if use_wcs:
        wcs = se_im.get_wcs_jacobian(x, y)
    else:
        wcs = galsim.PixelScale(1.0)

    psf_im = se_im.get_psf_image(x, y)
    cen = (psf_im.shape[0] - 1) / 2

    # check mean (x, y) to make sure it is not the center
    _y, _x = np.mgrid[:psf_im.shape[0], :psf_im.shape[1]]
    xbar = np.mean((_x - cen) * psf_im) / np.mean(psf_im)
    ybar = np.mean((_y - cen) * psf_im) / np.mean(psf_im)
    # PSFEx is not exactly centered, so the tolerance here is bigger
    assert np.abs(xbar - dx) < 1e-1, xbar
    assert np.abs(ybar - dy) < 1e-1, ybar

    psf = psfex.getPSF(galsim.PositionD(
        x=x+wcs_pos_offset, y=y+wcs_pos_offset))
    true_psf_im = psf.drawImage(
        nx=psf_im.shape[0],
        ny=psf_im.shape[0],
        wcs=wcs,
        offset=galsim.PositionD(x=dx, y=dy),
        method='no_pixel',
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


def get_center_delta(x):
    return x - np.ceil(x-0.5)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
@pytest.mark.parametrize('eps_x', [
    -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75])
@pytest.mark.parametrize('eps_y', [
    -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75])
def test_se_image_psf_piff(se_image_data, eps_x, eps_y, wcs_pos_offset):
    x = 10 + eps_x
    y = 11 + eps_y

    dx = get_center_delta(x)
    dy = get_center_delta(y)

    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        noise_seed=10,
        mask_tape_bumps=False,
    )

    psf_im = se_im.get_psf_image(x, y)

    cen = (psf_im.shape[0] - 1) / 2

    # check mean (x, y) to make sure it is not the center
    _y, _x = np.mgrid[:psf_im.shape[0], :psf_im.shape[1]]
    xbar = np.mean((_x - cen) * psf_im) / np.mean(psf_im)
    ybar = np.mean((_y - cen) * psf_im) / np.mean(psf_im)

    # Piff is not exactly centered, so the tolerance here is bigger
    assert np.abs(xbar - dx) < 1e-1, 'x: %g xbar: %g dx: %g' % (x, xbar, dx)
    assert np.abs(ybar - dy) < 1e-1, ybar

    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    true_psf_im = psf_mod.draw(
        x=x+wcs_pos_offset, y=y+wcs_pos_offset, stamp_size=PIFF_STAMP_SIZE,
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)
