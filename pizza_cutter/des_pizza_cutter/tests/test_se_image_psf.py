import os

import numpy as np
from scipy.spatial import cKDTree
import fitsio
import galsim
import galsim.hsm
import piff

import pytest

from .._se_image import (
    SEImageSlice,
    PIFF_STAMP_SIZE,
    _check_point_in_bad_piff_model_mask,
)


def test_check_point_in_bad_piff_model_mask():
    xdim = 1024
    ydim = 512
    grid_size = 64
    bad_msk = np.zeros((ydim//grid_size, xdim//grid_size)).astype(bool)
    bad_msk[3, 2] = True

    assert not _check_point_in_bad_piff_model_mask(-10, -10, bad_msk, grid_size)
    assert not _check_point_in_bad_piff_model_mask(45454, 343243, bad_msk, grid_size)
    assert _check_point_in_bad_piff_model_mask(145, 200, bad_msk, grid_size)
    assert _check_point_in_bad_piff_model_mask(128, 192, bad_msk, grid_size)
    assert not _check_point_in_bad_piff_model_mask(127, 191, bad_msk, grid_size)


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
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
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
    dx = x - np.floor(x + 0.5)
    dy = y - np.floor(y + 0.5)

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=galsim.Gaussian(fwhm=0.8),
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
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
    dx = x - np.floor(x + 0.5)
    dy = y - np.floor(y + 0.5)

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psfex,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
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
    return x - np.floor(x+0.5)


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
    x = 101 + eps_x
    y = 111 + eps_y

    dx = get_center_delta(x)
    dy = get_center_delta(y)

    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0,
        psf_kwargs={"GI_COLOR": 0.61},
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    psf_im_cen = se_im.get_psf_image(np.floor(x+0.5), np.floor(y+0.5))
    _y, _x = np.mgrid[:psf_im_cen.shape[0], :psf_im_cen.shape[1]]
    xcen = np.mean(_x * psf_im_cen) / np.mean(psf_im_cen)
    ycen = np.mean(_y * psf_im_cen) / np.mean(psf_im_cen)

    # check mean (x, y) to make sure it is not the center
    psf_im = se_im.get_psf_image(x, y)
    _y, _x = np.mgrid[:psf_im.shape[0], :psf_im.shape[1]]
    xbar = np.mean((_x - xcen) * psf_im) / np.mean(psf_im)
    ybar = np.mean((_y - ycen) * psf_im) / np.mean(psf_im)

    # Piff is not exactly centered, so the tolerance here is bigger
    print("\ncenter offsets:", xbar, dx, ybar, dy)

    assert np.abs(xbar - dx) < 1e-1, 'x: %g xbar: %g dx: %g' % (x, xbar, dx)
    assert np.abs(ybar - dy) < 1e-1, 'y: %g ybar: %g dy: %g' % (y, ybar, dy)

    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    image = galsim.ImageD(
        PIFF_STAMP_SIZE,
        PIFF_STAMP_SIZE,
        wcs=se_im.get_wcs_jacobian(x, y),
    )
    true_psf_im = psf_mod.draw(
        x=x+wcs_pos_offset,
        y=y+wcs_pos_offset,
        image=image,
        center=True,
        offset=(x - np.floor(x+0.5), y - np.floor(y+0.5)),
        GI_COLOR=0.61,
        chipnum=se_image_data["source_info"]["ccdnum"],
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [1])
@pytest.mark.parametrize('eps_x', [-0.50])
@pytest.mark.parametrize('eps_y', [0.25])
def test_se_image_psf_piff_color(se_image_data, eps_x, eps_y, wcs_pos_offset):
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
        wcs_color=0.7,
        psf_kwargs={"GI_COLOR": 0.61},
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    psf_im_cen = se_im.get_psf_image(np.floor(x+0.5), np.floor(y+0.5))
    _y, _x = np.mgrid[:psf_im_cen.shape[0], :psf_im_cen.shape[1]]
    xcen = np.mean(_x * psf_im_cen) / np.mean(psf_im_cen)
    ycen = np.mean(_y * psf_im_cen) / np.mean(psf_im_cen)

    # check mean (x, y) to make sure it is not the center
    psf_im = se_im.get_psf_image(x, y)
    _y, _x = np.mgrid[:psf_im.shape[0], :psf_im.shape[1]]
    xbar = np.mean((_x - xcen) * psf_im) / np.mean(psf_im)
    ybar = np.mean((_y - ycen) * psf_im) / np.mean(psf_im)

    # Piff is not exactly centered, so the tolerance here is bigger
    assert np.abs(xbar - dx) < 1e-1, 'x: %g xbar: %g dx: %g' % (x, xbar, dx)
    assert np.abs(ybar - dy) < 1e-1, ybar

    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    image = galsim.ImageD(
        PIFF_STAMP_SIZE,
        PIFF_STAMP_SIZE,
        wcs=se_im.get_wcs_jacobian(x, y),
    )
    true_psf_im = psf_mod.draw(
        x=x+wcs_pos_offset,
        y=y+wcs_pos_offset,
        image=image,
        center=True,
        offset=(x - np.floor(x+0.5), y - np.floor(y+0.5)),
        chipnum=se_image_data['source_info']['ccdnum'],
        GI_COLOR=0.61,
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)

    # piff defaults to no color used for this test data
    not_true_psf_im = psf_mod.draw(
        x=x+wcs_pos_offset,
        y=y+wcs_pos_offset,
        stamp_size=psf_im.shape[0],
        chipnum=se_image_data['source_info']['ccdnum'],
        GI_COLOR=0.61,
    ).array
    not_true_psf_im /= np.sum(not_true_psf_im)
    assert not np.array_equal(psf_im, not_true_psf_im)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_psf_piff_hsm(se_image_data):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    wcs = psf_mod.wcs[se_image_data["source_info"]["ccdnum"]]
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=wcs,
        wcs_position_offset=1,
        wcs_color=1.6,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )

    cat = fitsio.read(se_image_data["source_info"]["piff_cat_path"], lower=True)
    hsm_cat = fitsio.read(se_image_data["source_info"]["piff_hsmcat_path"])

    ctree = cKDTree(np.array([cat["xwin_image"], cat["ywin_image"]]).T)
    lol = ctree.query_ball_point(np.array([hsm_cat["x"], hsm_cat["y"]]).T, 1e-4)
    inds = [lst[0] for lst in lol]
    cat = cat[inds]
    assert np.array_equal(cat["xwin_image"], hsm_cat["x"])
    assert np.array_equal(cat["ywin_image"], hsm_cat["y"])

    for i in range(len(cat)):
        if hsm_cat["reserve"][i] != 1:
            continue

        x = hsm_cat["x"][i]
        y = hsm_cat["y"][i]
        psf_im = se_im.get_psf_image(
            x-1, y-1,
            psf_kwargs={"GI_COLOR": cat["gi_color"][i]},
        )
        jac = se_im.get_wcs_jacobian(x-1, y-1)

        im = galsim.ImageD(psf_im, wcs=jac)

        res = galsim.hsm.FindAdaptiveMom(im)
        # this code is lifted from Piff under its license
        sigma = res.moments_sigma
        shape = res.observed_shape
        scale, shear, theta, flip = jac.getDecomposition()
        # Fix sigma
        sigma *= scale
        # Fix shear.  First the flip, if any.
        if flip:
            shape = galsim.Shear(g1=-shape.g1, g2=shape.g2)
        # Next the rotation
        shape = galsim.Shear(g=shape.g, beta=shape.beta + theta)
        # Finally the shear
        shape = shear + shape

        # this T is actually sigma due to a Piff bug
        assert np.allclose(sigma, hsm_cat["T_model"][i], atol=5e-4, rtol=0)
        shear_eps = 2e-3
        assert np.allclose(shape.g1, hsm_cat["g1_model"][i], atol=shear_eps, rtol=0)
        assert np.allclose(shape.g2, hsm_cat["g2_model"][i], atol=shear_eps, rtol=0)
