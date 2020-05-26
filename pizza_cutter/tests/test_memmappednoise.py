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


@pytest.mark.parametrize('bad_value', [0.0, -10])
def test_memmappednoise_fill(bad_value):
    weight = np.ones((100, 100)) * 0.0001
    weight[0:50, :] = bad_value
    ns = MemMappedNoiseImage(
        seed=10, weight=weight, sx=10, sy=10, fill_weight=1/9)

    assert np.std(ns[0:50, :]) < np.std(ns[50:, :])

    # the noise should be close to 3, but ofc it is noise so within 0.1 is fine
    assert np.abs(np.std(ns[0:50, :]) - 3) < 0.1


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


def test_memmappednoise_sizes_weird():
    weight = np.ones((100, 100)) * 0.0001
    ns = MemMappedNoiseImage(seed=10, weight=weight, sx=74, sy=10)

    ns1 = ns[0, 0]
    ns2 = ns[0, 0]
    assert ns1 == ns2

    ns1 = ns[0, 0:10]
    ns2 = ns[0, 0:10]
    assert np.all(ns1 == ns2)

    ns1 = ns[0:5, 0:93]
    ns2 = ns[0, 0:93]
    assert np.all(ns1[0, :] == ns2)

    # if we missed any elements, they would be exactly zero
    ns1 = ns[:, :]
    assert not np.any(ns1 == 0)
