import numpy as np
from ..wcs import wrap_ra_diff

import pytest


def test_wrap_dra_array():
    dra = np.array([-350, -170, 0, 350, 350 + 360*10, -350 - 360*10])
    ans = np.array([10, -170, 0, -10, -10, 10])
    assert np.allclose(wrap_ra_diff(dra), ans)


@pytest.mark.parametrize("dra,ans", [
    (-350, 10),
    (-170, -170),
    (0, 0),
    (350, -10),
    (350 + 360*10, -10),
    (-350 - 360*10, 10),
])
def test_wrap_dra_scalar(dra, ans):
    assert np.allclose(wrap_ra_diff(dra), ans)


def test_wrap_dra_scalar_nan_inf():
    assert np.isnan(wrap_ra_diff(np.nan))
    assert np.isinf(wrap_ra_diff(np.inf))


def test_wrap_dra_array_nan_inf():
    dra = np.array([np.nan, np.inf, -350, -170, 0, 350, 350 + 360*10, -350 - 360*10])
    ans = np.array([np.nan, np.inf, 10, -170, 0, -10, -10, 10])
    msk = np.isfinite(dra)
    assert np.allclose(wrap_ra_diff(dra[msk]), ans[msk])
    assert np.isnan(ans[0])
    assert np.isinf(ans[1])
