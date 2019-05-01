import numpy as np
import pytest

import galsim

from .._se_image import SEImageSlice


@pytest.mark.parametrize('x,y', [
    (np.ones(10), 10),
    (np.ones((10, 10)), 10)])
def test_se_image_wcs_jacobian_array(se_image_data, x, y):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(x, y)

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(y, x)

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(x, y)

    with pytest.raises(AssertionError):
        se_im.get_wcs_jacobian(y, x)


def test_se_image_wcs_jacobian_esutil(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        jac = se_im.get_wcs_jacobian(x, y)
        tup = se_image_data['eu_wcs'].get_jacobian(x+1, y+1)
        assert jac.dudx == tup[0]
        assert jac.dudy == tup[1]
        assert jac.dvdx == tup[2]
        assert jac.dvdy == tup[3]


def test_se_image_wcs_jacobian_galsim(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['gs_wcs'], noise_seed=10)

    rng = np.random.RandomState(seed=10)
    for _ in range(10):
        x = rng.uniform() * 2048
        y = rng.uniform() * 4096
        jac = se_im.get_wcs_jacobian(x, y)
        gs_jac = se_image_data['gs_wcs'].local(
            galsim.PositionD(x=x+1, y=y+1))
        assert jac.dudx == gs_jac.dudx
        assert jac.dudy == gs_jac.dudy
        assert jac.dvdx == gs_jac.dvdx
        assert jac.dvdy == gs_jac.dvdy
