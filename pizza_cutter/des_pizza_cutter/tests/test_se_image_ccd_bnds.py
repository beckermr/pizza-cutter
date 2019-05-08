import pytest
from meds.bounds import Bounds

from .._se_image import SEImageSlice


def test_se_image_ccd_bnds_in(se_image_data):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    in_bnds = Bounds(10, 20, 50, 60)
    assert se_im.ccd_contains_bounds(in_bnds)


@pytest.mark.parametrize('in_bnds', [
    Bounds(0, 20, 50, 60),
    Bounds(10, 4095, 50, 60),
    Bounds(10, 20, 0, 60),
    Bounds(10, 20, 50, 2047)])
def test_se_image_ccd_bnds_in_edge(se_image_data, in_bnds):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    assert se_im.ccd_contains_bounds(in_bnds)


@pytest.mark.parametrize('over_bnds', [
    Bounds(-10, 20, 50, 60),
    Bounds(10, 8000, 50, 60),
    Bounds(10, 20, -50, 60),
    Bounds(10, 20, 50, 6000),
    Bounds(-10, 8000, -50, 6000)])
def test_se_image_ccd_bnds_over(se_image_data, over_bnds):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    assert not se_im.ccd_contains_bounds(over_bnds)


@pytest.mark.parametrize('out_bnds', [
    Bounds(-20, -10, -60, -50),
    Bounds(-20, -10, 50000, 60000),
    Bounds(10000, 20000, -60, -50),
    Bounds(10000, 20000, 50000, 60000)])
def test_se_image_ccd_bnds_out(se_image_data, out_bnds):
    se_im = SEImageSlice(
        source_info=None, psf_model=None,
        wcs=se_image_data['eu_wcs'], noise_seed=10)

    assert not se_im.ccd_contains_bounds(out_bnds)