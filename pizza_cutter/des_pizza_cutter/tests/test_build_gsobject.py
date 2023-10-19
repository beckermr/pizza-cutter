import galsim
import galsim.config
import numpy as np

from .._load_info import _build_gsobject


def test_build_gsobject_psf_dict():
    psf_dict = {'type': 'Gaussian', 'fwhm': 0.9}
    psf = _build_gsobject(psf_dict)
    assert isinstance(psf, galsim.Gaussian)
    assert psf.fwhm == 0.9


def test_build_gsobject_evalrepr():
    g = galsim.Gaussian(fwhm=1.0).drawImage(nx=25, ny=25, scale=0.263)
    ii = galsim.InterpolatedImage(g, x_interpolant='lanczos15')
    with np.printoptions(threshold=np.inf, precision=32):
        r = repr(ii)
    cfg = {
        "type": "Eval",
        "str": r.replace("array(", "np.array("),
    }
    rii = _build_gsobject(cfg)
    assert rii == ii
