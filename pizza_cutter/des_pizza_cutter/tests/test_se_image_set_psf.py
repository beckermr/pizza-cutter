import numpy as np
import pytest

import galsim

from .._se_image import SEImageSlice


@pytest.mark.parametrize('eps_x', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize('eps_y', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
def test_se_image_set_psf(se_image_data, eps_x, eps_y):
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=galsim.Gaussian(fwhm=0.8),
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    # do the WCS inversion in the function (and it not being exact),
    # we take the coordinates through a round trip to make sure things are
    # internally consistent
    x = 50 + eps_x
    y = 60 + eps_y
    ra, dec = se_im.image2sky(x, y)
    x, y = se_im.sky2image(ra, dec)
    assert np.allclose(x - eps_x, 50)
    assert np.allclose(y - eps_y, 60)

    dx = x - int(x + 0.5)
    dy = y - int(y + 0.5)
    x_start = int(x + 0.5) - 9
    y_start = int(y + 0.5) - 9

    se_im.set_psf(ra, dec)

    true_psf_im = galsim.Gaussian(fwhm=0.8).drawImage(
        nx=19,
        ny=19,
        wcs=se_im.get_wcs_jacobian(x, y),
        offset=galsim.PositionD(x=dx, y=dy)
    ).array
    true_psf_im /= np.sum(true_psf_im)

    assert np.array_equal(se_im.psf, true_psf_im)
    assert se_im.psf_box_size == 19
    assert se_im.psf_x_start == x_start
    assert se_im.psf_y_start == y_start
