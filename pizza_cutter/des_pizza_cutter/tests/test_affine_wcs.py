import numpy as np

import pytest

from .._affine_wcs import AffineWCS


def test_affine_wcs_is_celestial():
    wcs = AffineWCS(
        dudx=0.5,
        dudy=0.7,
        dvdx=-0.1,
        dvdy=-2.5,
        x0=56,
        y0=-1345
    )
    assert not wcs.is_celestial()


def test_affine_wcs_values():
    wcs = AffineWCS(
        dudx=0.5,
        dudy=0.7,
        dvdx=-0.1,
        dvdy=-2.5,
        x0=56,
        y0=-1345
    )
    u, v = wcs.image2sky(0.4, 0.8)
    assert u == 0.5 * (0.4 - 56) + 0.7 * (0.8 + 1345)
    assert v == -0.1 * (0.4 - 56) - 2.5 * (0.8 + 1345)


def test_affine_wcs_jacobian():
    wcs = AffineWCS(
        dudx=0.54,
        dudy=0.384257,
        dvdx=-0.3209411,
        dvdy=-3913431,
        x0=0,
        y0=0
    )
    jac = wcs.get_jacobian(None, None)
    assert jac[0] == wcs.dudx
    assert jac[1] == wcs.dudy
    assert jac[2] == wcs.dvdx
    assert jac[3] == wcs.dvdy


def test_affine_wcs_inverse():
    rng = np.random.RandomState(seed=19)
    for _ in range(10):
        wcs = AffineWCS(
            dudx=rng.uniform()-0.5,
            dudy=rng.uniform()-0.5,
            dvdx=rng.uniform()-0.5,
            dvdy=rng.uniform()-0.5,
            x0=rng.uniform()-0.5,
            y0=rng.uniform()-0.5
        )

        x, y = rng.normal(size=10), rng.normal(size=10)
        u, v = wcs.image2sky(x, y)
        _x, _y = wcs.sky2image(u, v)

        assert np.allclose(x, _x)
        assert np.allclose(y, _y)


def test_affine_wcs_zero_det_raises():
    with pytest.raises(ValueError):
        AffineWCS(
            dudx=0,
            dudy=0,
            dvdx=0,
            dvdy=0,
            x0=0,
            y0=0)
