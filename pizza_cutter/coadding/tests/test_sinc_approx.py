import time
import numpy as np
from numba import njit

from .._sinc import sinc, sinc_pade


def test_sinc_pade_correct():
    rng = np.random.RandomState(seed=32485)
    for _ in range(100000):
        x = rng.uniform() * 3
        # the olerances here were set iteratively to be as low as I could get
        # them - this is a regression test more than anything else
        assert np.allclose(sinc_pade(x), np.sinc(x), atol=1e-6, rtol=8e-7), (
            np.abs(sinc_pade(x) - np.sinc(x)), 1.0 - sinc_pade(x) / np.sinc(x)
        )

    assert np.allclose(sinc_pade(0), 1.0)
    assert np.allclose(sinc_pade(1), 0.0)
    assert np.allclose(sinc_pade(2), 0.0)
    assert np.allclose(sinc_pade(3), 0.0)


@njit(fastmath=True)
def _vec_sinc(x, y):
    n = x.shape[0]
    for i in range(n):
        y[i] = sinc(x[i])


@njit(fastmath=True)
def _vec_sinc_pade(x, y):
    n = x.shape[0]
    for i in range(n):
        y[i] = sinc_pade(x[i])


def test_sinc_pade_fast():
    rng = np.random.RandomState(seed=32485)
    x = rng.uniform(size=10_000_000) * 3
    y = np.zeros_like(x)

    # get it tp compile
    for _ in range(3):
        _vec_sinc(x, y)
        _vec_sinc_pade(x, y)

    t0 = time.time()
    for _ in range(10):
        _vec_sinc(x, y)
    time_sinc = time.time() - t0

    t0 = time.time()
    for _ in range(10):
        _vec_sinc_pade(x, y)
    time_sinc_pade = time.time() - t0

    print("\ntime sinc|sinc_pade: %s|%s" % (time_sinc, time_sinc_pade))
    assert time_sinc_pade < time_sinc * 0.70
