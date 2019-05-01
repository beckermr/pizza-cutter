import numpy as np
import pytest

from ..memmappednoise import MemMappedNoiseImage


def test_memmappednoise_smoke():
    weight = np.ones((100, 100)) * 0.0001
    ns = MemMappedNoiseImage(seed=10, weight=weight, sx=10, sy=10)

    ns1 = ns[0, 0]
    ns2 = ns[0, 0]
    assert ns1 == ns2

    ns1 = ns[0, 0:10]
    ns2 = ns[0, 0:10]
    assert np.all(ns1 == ns2)

    ns1 = ns[0:5, 0:93]
    ns2 = ns[0, 0:93]
    assert np.all(ns1[0, :] == ns2)


def test_memmappednoise_seeding():
    weight = np.ones((100, 100)) * 0.0001
    ns1 = MemMappedNoiseImage(seed=10, weight=weight, sx=10, sy=10)

    weight = np.ones((100, 100)) * 0.0001
    ns2 = MemMappedNoiseImage(seed=10, weight=weight, sx=10, sy=10)

    assert ns1[0, 0] == ns2[0, 0]
    assert np.all(ns1[0, 0:10] == ns2[0, 0:10])
    assert np.all(ns1[0:5, 0:24] == ns2[0:5, 0:24])

    weight = np.ones((100, 100)) * 0.0001
    ns2 = MemMappedNoiseImage(seed=20, weight=weight, sx=10, sy=10)

    assert ns1[0, 0] != ns2[0, 0]
    assert not np.all(ns1[0, 0:10] == ns2[0, 0:10])
    assert not np.all(ns1[0:5, 0:24] == ns2[0:5, 0:24])


def test_memmappednoise_sizes_same():
    weight = np.ones((100, 100)) * 0.0001
    ns = MemMappedNoiseImage(seed=10, weight=weight, sx=100, sy=100)

    ns1 = ns[0, 0]
    ns2 = ns[0, 0]
    assert ns1 == ns2

    ns1 = ns[0, 0:10]
    ns2 = ns[0, 0:10]
    assert np.all(ns1 == ns2)

    ns1 = ns[0:5, 0:93]
    ns2 = ns[0, 0:93]
    assert np.all(ns1[0, :] == ns2)


def test_memmappednoise_sizes_different():
    weight = np.ones((100, 100)) * 0.0001
    ns = MemMappedNoiseImage(seed=10, weight=weight, sx=25, sy=10)

    ns1 = ns[0, 0]
    ns2 = ns[0, 0]
    assert ns1 == ns2

    ns1 = ns[0, 0:10]
    ns2 = ns[0, 0:10]
    assert np.all(ns1 == ns2)

    ns1 = ns[0:5, 0:93]
    ns2 = ns[0, 0:93]
    assert np.all(ns1[0, :] == ns2)


def test_memmappednoise_sizes_raises():
    weight = np.ones((100, 100)) * 0.0001
    with pytest.raises(AssertionError):
        MemMappedNoiseImage(seed=10, weight=weight, sx=10, sy=13)

    with pytest.raises(AssertionError):
        MemMappedNoiseImage(seed=10, weight=weight, sx=13, sy=10)
