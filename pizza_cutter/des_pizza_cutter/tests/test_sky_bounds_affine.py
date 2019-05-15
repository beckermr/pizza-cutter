import pytest

import numpy as np

from .._sky_bounds import get_rough_sky_bounds
from .._affine_wcs import AffineWCS


@pytest.fixture
def bnds_data():
    wcs = AffineWCS(
        dudx=0.263,
        dudy=0,
        dvdx=0,
        dvdy=0.263,
        x0=10,
        y0=15)
    position_offset = 1
    sky_bnds, ra_ccd, dec_ccd = get_rough_sky_bounds(
        im_shape=(512, 256),
        wcs=wcs,
        position_offset=position_offset,
        bounds_buffer_uv=16.0,
        n_grid=4,
        celestial=False)

    return wcs, position_offset, sky_bnds, ra_ccd, dec_ccd


def test_get_rough_sky_bounds_affine_smoke(bnds_data):
    wcs, position_offset, sky_bnds, ra_ccd, dec_ccd = bnds_data
    ncol, nrow = 256, 512
    row, col = np.mgrid[0:nrow+64:64, 0:ncol+64:64]
    row = row.ravel()
    col = col.ravel()

    ra, dec = wcs.image2sky(
        x=col + position_offset,
        y=row + position_offset)
    u, v = ra - ra_ccd, dec - dec_ccd
    assert np.all(sky_bnds.contains_points(u, v))

    # dither things a bit too
    for col_dither in [-0.51, 0.51]:
        for row_dither in [-0.51, 0.52]:
            ra, dec = wcs.image2sky(
                x=col + position_offset + col_dither,
                y=row + position_offset + row_dither)
            u, v = ra - ra_ccd, dec - dec_ccd
            assert np.all(sky_bnds.contains_points(u, v))


@pytest.mark.parametrize(
    'offset,outside',
    [(45, False),
     (75, True)])
def test_get_rough_sky_bounds_affine_edge_buffer(bnds_data, offset, outside):
    # the buffer above is 16 arcsec
    # thus we want 16 / 0.263 ~ 61 pixels
    # so we test inside at 45 and outside at 75

    wcs, position_offset, sky_bnds, ra_ccd, dec_ccd = bnds_data

    ncol, nrow = 265, 512
    row_t = np.linspace(0, nrow, (nrow + 64) // 64)
    col_t = np.ones(nrow // 64 + 1) * 0
    row_b = np.linspace(0, nrow, (nrow + 64) // 64)
    col_b = np.ones(nrow // 64 + 1) * (ncol - 1)

    row_l = np.ones(ncol // 64 + 1) * 0
    col_l = np.linspace(0, ncol, (ncol + 64) // 64)
    row_r = np.ones(ncol // 64 + 1) * (nrow - 1)
    col_r = np.linspace(0, ncol, (ncol + 64) // 64)

    def _test_it(row, col, col_dither, row_dither, outside=False):
        ra, dec = wcs.image2sky(
            x=col + position_offset + col_dither,
            y=row + position_offset + row_dither)
        u, v = ra - ra_ccd, dec - dec_ccd
        if outside:
            assert np.all(~sky_bnds.contains_points(u, v))
        else:
            assert np.all(sky_bnds.contains_points(u, v))

    _test_it(row_t, col_t, -offset, 0, outside=outside)
    _test_it(row_b, col_b, offset, 0, outside=outside)
    _test_it(row_l, col_l, 0, -offset, outside=outside)
    _test_it(row_r, col_r, 0, offset, outside=outside)
