import galsim.config

from .._load_info import _build_gsobject


def test_build_gsobject_psf_dict():
    psf_dict = {'type': 'Gaussian', 'fwhm': 0.9}
    psf = _build_gsobject(psf_dict)
    assert isinstance(psf, galsim.Gaussian)
    assert psf.fwhm == 0.9
