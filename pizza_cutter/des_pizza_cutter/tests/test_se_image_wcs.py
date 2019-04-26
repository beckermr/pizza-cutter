import numpy as np
import pytest

import galsim

from .._se_image import SEImageSlice


@pytest.mark.parametrize('x,y', [
    (np.ones(10), 10),
    (np.ones((10, 10)), 10)])
def test_se_image_wcs_array_shape(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], random_state=10)

    with pytest.raises(AssertionError):
        se_im.image2sky(x, y)

    with pytest.raises(AssertionError):
        se_im.image2sky(y, x)

    with pytest.raises(AssertionError):
        se_im.sky2image(x, y)

    with pytest.raises(AssertionError):
        se_im.sky2image(y, x)


def test_se_image_wcs_esutil(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], random_state=10)

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        ra, dec = se_im.image2sky(x, y)
        eu_ra, eu_dec = se_image_data['eu_wcs'].image2sky(x+1, y+1)
        assert np.allclose(ra, eu_ra)
        assert np.allclose(dec, eu_dec)

        x, y = se_im.sky2image(ra, dec)
        eu_x, eu_y = se_image_data['eu_wcs'].sky2image(ra, dec)
        assert np.allclose(x, eu_x - 1)
        assert np.allclose(y, eu_y - 1)


def test_se_image_wcs_esutil_array(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], random_state=10)

    x = np.arange(10)
    y = np.arange(10)
    ra, dec = se_im.image2sky(x, y)
    assert ra.shape == x.shape
    assert dec.shape == x.shape

    eu_ra, eu_dec = se_image_data['eu_wcs'].image2sky(x+1, y+1)
    assert np.allclose(ra, eu_ra)
    assert np.allclose(dec, eu_dec)

    x, y = se_im.sky2image(ra, dec)
    assert x.shape == ra.shape
    assert y.shape == ra.shape

    eu_x, eu_y = se_image_data['eu_wcs'].sky2image(ra, dec)
    assert np.allclose(x, eu_x - 1)
    assert np.allclose(y, eu_y - 1)


def test_se_image_wcs_galsim(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['gs_wcs'], random_state=10)

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        ra, dec = se_im.image2sky(x, y)
        pos = se_image_data['gs_wcs'].toWorld(
            galsim.PositionD(x=x+1, y=y+1))
        assert np.allclose(ra, pos.ra / galsim.degrees)
        assert np.allclose(dec, pos.dec / galsim.degrees)

        x, y = se_im.sky2image(ra, dec)
        wpos = galsim.CelestialCoord(
            ra=ra * galsim.degrees,
            dec=dec * galsim.degrees)
        pos = se_image_data['gs_wcs'].toImage(wpos)
        assert np.allclose(x, pos.x - 1)
        assert np.allclose(y, pos.y - 1)


def test_se_image_wcs_galsim_array(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['gs_wcs'], random_state=10)

    x = np.arange(2)
    y = np.arange(2) + 100

    ra, dec = se_im.image2sky(x, y)
    x_out, y_out = se_im.sky2image(ra, dec)

    for i in range(2):
        pos = se_image_data['gs_wcs'].toWorld(
            galsim.PositionD(x=x[i]+1, y=y[i]+1))
        assert np.allclose(ra[i], pos.ra / galsim.degrees)
        assert np.allclose(dec[i], pos.dec / galsim.degrees)

        wpos = galsim.CelestialCoord(
            ra=ra[i] * galsim.degrees,
            dec=dec[i] * galsim.degrees)
        pos = se_image_data['gs_wcs'].toImage(wpos)
        assert np.allclose(x[i], pos.x - 1)
        assert np.allclose(y[i], pos.y - 1)
