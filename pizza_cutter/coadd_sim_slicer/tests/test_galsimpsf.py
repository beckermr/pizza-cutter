import numpy as np
import galsim

from ..galsim_psf import GalSimPSF

WCS_HEADER = dict(
    naxis1=100,
    naxis2=100,
    ctype1='RA---TAN',
    ctype2='DEC--TAN',
    crpix1=50.5,
    crpix2=50.5,
    cd1_1=-7.305555555556E-05,
    cd1_2=0.0,
    cd2_1=0.0,
    cd2_2=7.305555555556E-05,
    cunit1='deg     ',
    cunit2='deg     ',
    crval1=321.417528,
    crval2=1.444444)
WCS = galsim.FitsWCS(header={k.upper(): v for k, v in WCS_HEADER.items()})


def test_galsim_psf_dict():
    psf_dict = {'type': 'Gaussian', 'fwhm': 0.9}
    gspsf = GalSimPSF(psf_dict, WCS, npix=27)
    row = 23.0
    col = 57.0

    # the center is (27 - 1) / 2
    assert np.array_equal(gspsf.get_center(row, col), [13, 13])

    # we have to convert to pixels
    ps = np.sqrt(WCS.pixelArea(image_pos=galsim.PositionD(row+1, col+1)))
    assert np.allclose(gspsf.get_sigma(row, col), 0.9 / ps / 2.35482004503)

    img = gspsf.get_rec(row, col)
    assert img.shape == (27, 27)
    assert np.allclose(img.sum(), 1.0)
    loc = np.unravel_index(np.argmax(img), (27, 27))
    assert np.array_equal(loc, [13, 13])


def test_galsim_psf_object():
    psf = galsim.Gaussian(fwhm=0.9)
    gspsf = GalSimPSF(psf, WCS, npix=27)
    row = 23.0
    col = 57.0

    # the center is (27 - 1) / 2
    assert np.array_equal(gspsf.get_center(row, col), [13, 13])

    # we have to convert to pixels
    ps = np.sqrt(WCS.pixelArea(image_pos=galsim.PositionD(row+1, col+1)))
    assert np.allclose(gspsf.get_sigma(row, col), 0.9 / ps / 2.35482004503)

    img = gspsf.get_rec(row, col)
    assert img.shape == (27, 27)
    assert np.allclose(img.sum(), 1.0)
    loc = np.unravel_index(np.argmax(img), (27, 27))
    assert np.array_equal(loc, [13, 13])


def test_galsim_psf_dict_eval():
    psf_dict = {
        'type': 'Gaussian',
        'fwhm':
            ('$0.9 + '
             '0.5 * np.sqrt((row - row_cen)**2 + (col - col_cen)**2) / 50')}
    cen = (100 - 1) / 2
    gspsf = GalSimPSF(
        psf_dict,
        WCS,
        npix=27,
        eval_locals={'row_cen': cen, 'col_cen': cen})
    row = 23.0
    col = 57.0

    # the center is (27 - 1) / 2
    assert np.array_equal(gspsf.get_center(row, col), [13, 13])

    # we have to convert to pixels
    ps = np.sqrt(WCS.pixelArea(image_pos=galsim.PositionD(row+1, col+1)))
    fwhm = 0.9 + 0.5 * np.sqrt((row - cen)**2 + (col - cen)**2) / 50
    assert np.allclose(gspsf.get_sigma(row, col), fwhm / ps / 2.35482004503)

    img = gspsf.get_rec(row, col)
    assert img.shape == (27, 27)
    assert np.allclose(img.sum(), 1.0)
    loc = np.unravel_index(np.argmax(img), (27, 27))
    assert np.array_equal(loc, [13, 13])
