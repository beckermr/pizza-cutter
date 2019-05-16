import numpy as np
import pytest

from .._lanczos import lanczos_resample


def test_lanczos1_resample_smoke():
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    row = 5.5
    col = 7.5
    val, _ = lanczos_resample(
        im,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64),
        a=1)

    # compute val by hand to compare
    row_s = 5  # int(np.floor(row)) - a + 1
    row_e = 6  # int(np.floor(row)) + a
    col_s = 7  # int(np.floor(col)) - a + 1
    col_e = 8  # int(np.floor(col)) + a
    true_val = 0.0
    for r in range(row_s, row_e+1):
        dr = row - r
        for c in range(col_s, col_e+1):
            dc = col - c

            true_val += (
                im[r, c] *
                np.sinc(dr) * np.sinc(dr/1) *
                np.sinc(dc) * np.sinc(dc/1))

    assert np.allclose(val, true_val)


def test_lanczos2_resample_smoke():
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    row = 5.5
    col = 7.5
    val, _ = lanczos_resample(
        im,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64),
        a=2)

    # compute val by hand to compare
    row_s = 4  # int(np.floor(row)) - a + 1
    row_e = 7  # int(np.floor(row)) + a
    col_s = 6  # int(np.floor(col)) - a + 1
    col_e = 9  # int(np.floor(col)) + a
    true_val = 0.0
    for r in range(row_s, row_e+1):
        dr = row - r
        for c in range(col_s, col_e+1):
            dc = col - c

            true_val += (
                im[r, c] *
                np.sinc(dr) * np.sinc(dr/2) *
                np.sinc(dc) * np.sinc(dc/2))

    assert np.allclose(val, true_val)


def test_lanczos3_resample_smoke():
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    row = 5.5
    col = 7.5
    val, _ = lanczos_resample(
        im,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64),
        a=3)

    # compute val by hand to compare
    row_s = 3  # int(np.floor(row)) - a + 1
    row_e = 8  # int(np.floor(row)) + a
    col_s = 5  # int(np.floor(col)) - a + 1
    col_e = 10  # int(np.floor(col)) + a
    true_val = 0.0
    for r in range(row_s, row_e+1):
        dr = row - r
        for c in range(col_s, col_e+1):
            dc = col - c

            true_val += (
                im[r, c] *
                np.sinc(dr) * np.sinc(dr/3) *
                np.sinc(dc) * np.sinc(dc/3))

    assert np.allclose(val, true_val)


@pytest.mark.parametrize('a', [1, 2, 3, 4, 5])
def test_lanczos_resample_interp_grid(a):
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    for row in range(11):
        for col in range(25):
            val, _ = lanczos_resample(
                im,
                np.array([row], dtype=np.float64),
                np.array([col], dtype=np.float64),
                a=a)
            assert np.allclose(val, im[row, col])


@pytest.mark.parametrize(
    'row, col', [
        # clearly bad
        (-10, 50),

        # one is ok
        (10, 50),
        (-10, 5),

        # just on the edge
        (-3.00001, 5),
        (13, 5),
        (10, -3.00001),
        (10, 27),

        # both on edge
        (13, 27),
        (-3.0001, -3.0001),
        (13, -3.0001),
        (-3.0001, 27)])
def test_lanczos_resample_out_of_bounds(row, col):
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    val, edge = lanczos_resample(
        im,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64),
        a=3)
    assert np.all(edge)


@pytest.mark.parametrize(
    'row, col', [
        # clearly good
        (10, 5),

        # just inside the edge
        (-3, 5),
        (12, 5),
        (10, -3),
        (10, 26),

        # both inside the edge
        (12, 26),
        (-3, -3),
        (12, -3),
        (-3, 26)])
def test_lanczos_resample_in_bounds(row, col):
    rng = np.random.RandomState(seed=10)
    im = rng.normal(size=(11, 25))

    val, edge = lanczos_resample(
        im,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64),
        a=3)
    assert np.all(~edge)
