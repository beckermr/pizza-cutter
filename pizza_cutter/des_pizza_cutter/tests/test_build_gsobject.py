import galsim.config

from .._load_info import _build_gsobject


def test_build_gsobject_psf_dict():
    psf_dict = {'type': 'Gaussian', 'fwhm': 0.9}
    psf = _build_gsobject(psf_dict)
    assert isinstance(psf, galsim.Gaussian)
    assert psf.fwhm == 0.9


def test_build_gsobject_psf_dict_eval():
    psf_dict = {
        'type': 'Gaussian',
        'fwhm': '$0.9 + 0.5'}
    psf = _build_gsobject(psf_dict)
    assert isinstance(psf, galsim.Gaussian)
    assert psf.fwhm == 1.4
