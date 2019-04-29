import numpy as np
import pytest

import galsim

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

    psf_im = se_im.get_psf_image(10.1, 11.5)
    ind = np.argmax(psf_im)
    cen = (psf_im.shape[0] - 1) / 2
    assert ind != psf_im.shape[1]*cen + cen

    true_psf_im = galsim.Gaussian(fwhm=0.8).drawImage(
        nx=19,
        ny=19,
        wcs=se_im.get_wcs_jacobian(10.1, 11.5),
        offset=galsim.PositionD(x=0.1, y=0.5)
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
