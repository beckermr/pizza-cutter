import numpy as np
import pytest

from .._se_image import SEImageSlice


@pytest.mark.parametrize('x,y', [
    (np.ones(10), 10),
    (np.ones((10, 10)), 10)])
def test_se_image_sky_bnds_array_shape(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], random_state=10)

    with pytest.raises(AssertionError):
        se_im.image2sky(x, y)

    with pytest.raises(AssertionError):
        se_im.contains_radec(y, x)


def test_se_image_sky_bnds_array(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], random_state=10)

    ra = se_im._ra_ccd * np.ones(10)
    dec = se_im._dec_ccd * np.ones(10)
    dec[0] += 15.0  # not in the CCD

    msk = se_im.contains_radec(ra, dec)
    assert not msk[0]
    assert np.all(msk[1:])


def test_se_image_sky_bnds_scalar(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], random_state=10)

    ra = se_im._ra_ccd
    dec = se_im._dec_ccd
    msk = se_im.contains_radec(ra, dec)
    assert msk
    assert msk is True

    dec += 15.0  # not in the CCD
    msk = se_im.contains_radec(ra, dec)
    assert not msk
    assert msk is False
