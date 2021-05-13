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
