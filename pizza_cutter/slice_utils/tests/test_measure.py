import numpy as np
import galsim

from ..measure import measure_fwhm


def test_measure_fwhm():
    g = galsim.Gaussian(fwhm=2)
    im = g.drawImage(nx=512, ny=512, scale=0.05).array

    fwhm = measure_fwhm(im)
    assert np.allclose(fwhm * 0.05, 2, atol=1e-1, rtol=0)
