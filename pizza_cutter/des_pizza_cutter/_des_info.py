import os

import esutil as eu
import galsim
import galsim.des
import fitsio

import pixmappy
import galsim.des
import desmeds

from ._constants import MAGZP_REF, POSITION_OFFSET
from ._piff_tools import load_piff_from_image_path


def get_des_y3_coadd_tile_info(*, tilename, band, campaign, medsconf):
    """Read the coadd tile info, load WCS info, and load PSF info for
    the DES Y3 DESDM layout.

    Parameters
    ----------
    tilename : str
        The name of the coadd tile.
    band : str
        The band as a single letter (e.g., 'r').
    campaign : str
        The coadd DESDM campaign (e.g., 'Y3A1_COADD')
    medsconf : str
        The MEDS version. This string is used to find where the source
        images are located

    Returns
    -------
    info : dict
        A dictionary with at least the following keys:

            'wcs' : the coadd `esutil.wcsutil.WCS` object
            'galsim_wcs' : the coadd `galsim.FitsWCS` object
            'position_offset' : the offset to add to zero-indexed image
                coordinates to get transform them to the convention assumed
                by the WCS.
            'src_info' : list of dicts for the SE sources
            'image_path' : the path to the FITS file with the coadd image
            'image_ext' : the name of the FITS extension with the coadd image
            'weight_path' : the path to the FITS file with the coadd weight map
            'weight_ext' : the name of the FITS extension with the coadd weight
                map
            'bmask_path' : the path to the FITS file with the coadd bit mask
            'bmask_ext' : the name of the FITS extension with the coadd bit
                mask
            'seg_path' : the path to the FITS file with the coadd seg map
            'seg_ext' : the name of the FITS extension with the coadd seg map

        The dictionaries in the 'src_info' list have at least the
        following keys:

            'wcs' : the SE `esutil.wcsutil.WCS` object
            'galsim_wcs' : the SE `galsim.FitsWCS` object
            'pixmappy_wcs' : the SE pixmappy WCS solution
            'image_path' : the path to the FITS file with the SE image
            'image_ext' : the name of the FITS extension with the SE image
            'bkg_path' : the path to the FITS file with the SE background image
            'bkg_ext' : the name of the FITS extension with the SE background
                image
            'weight_path' : the path to the FITS file with the SE weight map
            'weight_ext' : the name of the FITS extension with the SE weight
                map
            'bmask_path' : the path to the FITS file with the SE bit mask
            'bmask_ext' : the name of the FITS extension with the SE bit mask
            'psfex_psf' : a galsim.des.DES_PSFEx object with the PSFEx
                PSF reconstruction.
            'piff_psf' : a piff.PSF object with the Piff PSF reconstruction.
            'scale' : a multiplicative factor to apply to the image
                (`*= scale`) and weight map (`/= scale**2`) for magnitude
                zero-point calibration.
    """

    coadd_srcs = desmeds.coaddsrc.CoaddSrc(
        medsconf,
        tilename,
        band,
        campaign=campaign,
    )

    coadd = desmeds.coaddinfo.Coadd(
        medsconf,
        tilename,
        band,
        campaign=campaign,
        sources=coadd_srcs,
    )

    info = coadd.get_info()
    info['wcs'] = eu.wcsutil.WCS(
        _munge_fits_header(fitsio.read_header(info['image_path'], ext='sci')))
    info['galsim_wcs'] = galsim.FitsWCS(info['image_path'])
    info['position_offset'] = POSITION_OFFSET

    info['image_ext'] = 'sci'

    info['weight_path'] = info['image_path']
    info['weight_ext'] = 'wgt'

    info['bmask_path'] = info['image_path']
    info['bmask_ext'] = 'msk'

    info['seg_ext'] = 'sci'

    # always true for the coadd
    info['magzp'] = MAGZP_REF
    info['scale'] = 1.0

    info['image_flags'] = 0  # TODO set this properly

    for ii in info['src_info']:
        ii['image_flags'] = 0

        ii['image_ext'] = 'sci'

        ii['weight_path'] = ii['image_path']
        ii['weight_ext'] = 'wgt'

        ii['bmask_path'] = ii['image_path']
        ii['bmask_ext'] = 'msk'

        ii['bkg_ext'] = 'sci'

        # wcs info
        ii['wcs'] = eu.wcsutil.WCS(
            _munge_fits_header(
                fitsio.read_header(ii['image_path'], ext='sci')))
        ii['galsim_wcs'] = galsim.FitsWCS(ii['image_path'])
        ii['position_offset'] = POSITION_OFFSET

        # psfex psf
        ii['psfex_psf'] = galsim.des.DES_PSFEx(ii['psf_path'])

        # piff.  TODO make the piff_run configurable
        piff_data = load_piff_from_image_path(
            image_path=ii['image_path'],
            piff_run='y3a1-v29',
        )

        ii['piff_path'] = piff_data['psf_path']
        ii['piff_psf'] = piff_data['psf']
        ii['image_flags'] |= piff_data['flags']

        # pixmappy we get from the psf object
        if ii['piff_psf'] is None:
            ii['pixmappy_wcs'] = None
        else:
            ii['pixmappy_wcs'] = ii['piff_psf'].wcs[0]
            assert isinstance(ii['pixmappy_wcs'], pixmappy.GalSimWCS), (
                "We did not find a pixmappy WCS object for this SE image!"
            )
            

        # image scale
        ii['scale'] = 10.0**(0.4*(MAGZP_REF - ii['magzp']))

    return info


def _munge_fits_header(hdr):
    dct = {}
    for k in hdr.keys():
        try:
            dct[k.lower()] = hdr[k]
        except Exception:
            pass
    return dct


def _get_piff_path(image_path):
    PIFF_DATA_DIR = os.environ['PIFF_DATA_DIR']

    img = os.path.basename(image_path)
    img = img.replace('immasked.fits.fz', 'piff.fits')
    num = str(int(img.split('_')[0][1:]))  # strip leading zeros...

    return os.path.join(
        PIFF_DATA_DIR,
        'y3a1-v29',
        num,
        img
    )
