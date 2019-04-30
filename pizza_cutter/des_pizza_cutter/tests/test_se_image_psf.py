import numpy as np
import pytest

import galsim
import piff

from .._se_image import SEImageSlice


@pytest.mark.parametrize('x,y', [
    (np.ones(10), 10),
    (np.ones((10, 10)), 10)])
def test_se_image_psf_array(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    with pytest.raises(AssertionError):
        se_im.get_psf_image(x, y)

    with pytest.raises(AssertionError):
        se_im.get_psf_image(y, x)

    with pytest.raises(AssertionError):
        se_im.get_psf_image(x, y)

    with pytest.raises(AssertionError):
        se_im.get_psf_image(y, x)


def test_se_image_psf_gsobject_offcenter(se_image_data):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=galsim.Gaussian(fwhm=0.8),
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    psf_im = se_im.get_psf_image(10.75, 11.75)
    ind = np.argmax(psf_im)
    cen = (psf_im.shape[0] - 1) / 2
    assert ind != psf_im.shape[1]*cen + cen

    true_psf_im = galsim.Gaussian(fwhm=0.8).drawImage(
        nx=19,
        ny=19,
        wcs=se_im.get_wcs_jacobian(10.75, 11.75),
        offset=galsim.PositionD(x=0.75, y=0.75)
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


def test_se_image_psf_gsobject_center(se_image_data):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=galsim.Gaussian(fwhm=0.8),
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    psf_im = se_im.get_psf_image(10, 11)

    ind = np.argmax(psf_im)
    cen = (psf_im.shape[0] - 1) / 2
    assert ind == psf_im.shape[1]*cen + cen

    true_psf_im = galsim.Gaussian(fwhm=0.8).drawImage(
        nx=19,
        ny=19,
        wcs=se_im.get_wcs_jacobian(10, 11),
        offset=galsim.PositionD(x=0, y=0)
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


@pytest.mark.parametrize('use_wcs', [False, True])
def test_se_image_psf_psfex_offcenter(se_image_data, use_wcs):
    if use_wcs:
        psfex = galsim.des.DES_PSFEx(
            se_image_data['source_info']['psf_path'],
            se_image_data['source_info']['image_path'],
            )
    else:
        psfex = galsim.des.DES_PSFEx(
            se_image_data['source_info']['psf_path'])

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psfex,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    if use_wcs:
        wcs = se_im.get_wcs_jacobian(10.5, 11.5)
    else:
        wcs = galsim.PixelScale(1.0)

    psf_im = se_im.get_psf_image(10.5, 11.5)
    ind = np.argmax(psf_im)
    cen = (psf_im.shape[0] - 1) / 2
    assert ind != psf_im.shape[1]*cen + cen

    psf = psfex.getPSF(galsim.PositionD(x=10.5+1, y=11.5+1))
    true_psf_im = psf.drawImage(
        nx=psf_im.shape[0],
        ny=psf_im.shape[0],
        wcs=wcs,
        offset=galsim.PositionD(x=0.5, y=0.5),
        method='no_pixel',
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


@pytest.mark.parametrize('use_wcs', [False, True])
def test_se_image_psf_psfex_center(se_image_data, use_wcs):
    if use_wcs:
        psfex = galsim.des.DES_PSFEx(
            se_image_data['source_info']['psf_path'],
            se_image_data['source_info']['image_path'],
            )
    else:
        psfex = galsim.des.DES_PSFEx(
            se_image_data['source_info']['psf_path'])

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psfex,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    if use_wcs:
        wcs = se_im.get_wcs_jacobian(10, 11)
    else:
        wcs = galsim.PixelScale(1.0)

    psf_im = se_im.get_psf_image(10, 11)
    ind = np.argmax(psf_im)
    cen = (psf_im.shape[0] - 1) / 2
    assert ind == psf_im.shape[1]*cen + cen

    psf = psfex.getPSF(galsim.PositionD(x=10+1, y=11+1))
    true_psf_im = psf.drawImage(
        nx=psf_im.shape[0],
        ny=psf_im.shape[0],
        wcs=wcs,
        offset=galsim.PositionD(x=0, y=0),
        method='no_pixel',
    ).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


def test_se_image_psf_piff_offcenter(se_image_data):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    psf_im = se_im.get_psf_image(10.75, 11.75)
    ind = np.argmax(psf_im)
    cen = (psf_im.shape[0] - 1) / 2
    assert ind != psf_im.shape[1]*cen + cen

    true_psf_im = psf_mod.draw(
        10.75, 11.75, stamp_size=17, offset=(0.75, 0.75)).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)


def test_se_image_psf_piff_center(se_image_data):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    psf_im = se_im.get_psf_image(10, 11)
    ind = np.argmax(psf_im)
    cen = (psf_im.shape[0] - 1) / 2
    assert ind == psf_im.shape[1]*cen + cen

    true_psf_im = psf_mod.draw(
        10, 11, stamp_size=17, offset=(0, 0)).array
    true_psf_im /= np.sum(true_psf_im)
    assert np.array_equal(psf_im, true_psf_im)
