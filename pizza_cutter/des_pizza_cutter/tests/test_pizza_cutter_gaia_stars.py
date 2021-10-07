import numpy as np

import pytest

from .._pizza_cutter import _build_gaia_star_mask


@pytest.mark.parametrize("symmetrize", [True, False])
def test_build_gaia_star_mask(symmetrize):
    gaia_stars = np.array(
        [(20, 10, 19.5), (39, 29, 20.1)],
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("phot_g_mean_mag", "f8"),
        ],
    )

    rad = 10.0**(np.poly1d([1.5e-3, -1.6e-01,  3.5e+00])(19.5))
    sarea = np.pi*rad**2/4
    if symmetrize:
        sarea *= 2
    area_frac = sarea / 400.0

    gmask = _build_gaia_star_mask(
        gaia_stars=gaia_stars,
        max_g_mag=20,
        poly_coeffs=[1.5e-3, -1.6e-01,  3.5e+00],
        start_col=20,
        start_row=10,
        box_size=20,
        symmetrize=symmetrize,
    )
    assert gmask[0, 0]
    assert not gmask[-1, -1]
    if symmetrize:
        assert gmask[-1, 0]
    else:
        assert not gmask[-1, 0]
    assert np.allclose(np.mean(gmask), area_frac, rtol=0, atol=4e-2)
