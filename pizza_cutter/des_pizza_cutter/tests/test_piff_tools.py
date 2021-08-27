import os
import pytest

from .._piff_tools import compute_piff_flags, get_piff_psf_info


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/.desservices.ini")),
    reason="no DESDM access",
)
def test_get_piff_psf_info():
    psf_info = get_piff_psf_info(
        expnum=232321,
        piff_campaign="Y6A1_PIFF",
    )
    import pprint
    pprint.pprint(psf_info)
    assert not any(i["ccdnum"] == 31 for i in psf_info.values())


def test_compute_piff_flags():
    # should be flags == 0
    piff_info = dict(
        desdm_flags=0,
        fwhm_cen=2,
        star_t_std=0.03,
        star_t_mean=0.5,
        nstar=55,
        exp_star_t_mean=0.55,
        exp_star_t_std=0.02,
    )
    flags = compute_piff_flags(
        piff_info=piff_info,
        max_fwhm_cen=3,
        min_nstar=25,
        max_exp_T_mean_fac=4,
        max_ccd_T_std_fac=0.3,
    )
    assert flags == 0

    # too few stars
    piff_info = dict(
        desdm_flags=0,
        fwhm_cen=2,
        star_t_std=0.03,
        star_t_mean=0.5,
        nstar=20,
        exp_star_t_mean=0.55,
        exp_star_t_std=0.02,
    )
    flags = compute_piff_flags(
        piff_info=piff_info,
        max_fwhm_cen=3,
        min_nstar=25,
        max_exp_T_mean_fac=4,
        max_ccd_T_std_fac=0.3,
    )
    assert (flags & 2**3) != 0

    # desdm flags and too few stars
    piff_info = dict(
        desdm_flags=1,
        fwhm_cen=2,
        star_t_std=0.03,
        star_t_mean=0.5,
        nstar=20,
        exp_star_t_mean=0.55,
        exp_star_t_std=0.02,
    )
    flags = compute_piff_flags(
        piff_info=piff_info,
        max_fwhm_cen=3,
        min_nstar=25,
        max_exp_T_mean_fac=4,
        max_ccd_T_std_fac=0.3,
    )
    assert (flags & 2**3) != 0
    assert (flags & 2**0) != 0

    # desdm flags, fwhm_cen, and too few stars
    piff_info = dict(
        desdm_flags=1,
        fwhm_cen=4,
        star_t_std=0.03,
        star_t_mean=0.5,
        nstar=20,
        exp_star_t_mean=0.55,
        exp_star_t_std=0.02,
    )
    flags = compute_piff_flags(
        piff_info=piff_info,
        max_fwhm_cen=3,
        min_nstar=25,
        max_exp_T_mean_fac=4,
        max_ccd_T_std_fac=0.3,
    )
    assert (flags & 2**0) != 0
    assert (flags & 2**1) != 0
    assert (flags & 2**3) != 0

    # desdm flags, fwhm_cen, CCD T variation, and too few stars
    piff_info = dict(
        desdm_flags=1,
        fwhm_cen=4,
        star_t_std=0.03,
        star_t_mean=0.5,
        nstar=20,
        exp_star_t_mean=0.55,
        exp_star_t_std=0.02,
    )
    flags = compute_piff_flags(
        piff_info=piff_info,
        max_fwhm_cen=3,
        min_nstar=25,
        max_exp_T_mean_fac=4,
        max_ccd_T_std_fac=0.001,
    )
    assert (flags & 2**0) != 0
    assert (flags & 2**1) != 0
    assert (flags & 2**2) != 0
    assert (flags & 2**3) != 0

    # desdm flags, fwhm_cen, CCD T variation, exp T outlier, and too few stars
    piff_info = dict(
        desdm_flags=1,
        fwhm_cen=4,
        star_t_std=0.03,
        star_t_mean=0.5,
        nstar=20,
        exp_star_t_mean=0.55,
        exp_star_t_std=0.02,
    )
    flags = compute_piff_flags(
        piff_info=piff_info,
        max_fwhm_cen=3,
        min_nstar=25,
        max_exp_T_mean_fac=0.001,
        max_ccd_T_std_fac=0.001,
    )
    assert (flags & 2**0) != 0
    assert (flags & 2**1) != 0
    assert (flags & 2**2) != 0
    assert (flags & 2**3) != 0
    assert (flags & 2**4) != 0
