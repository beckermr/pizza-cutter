import esutil as eu
import galsim.des
import fitsio
import logging

import pixmappy
import desmeds

from ._constants import MAGZP_REF, POSITION_OFFSET
from ._piff_tools import load_piff_from_image_path

logger = logging.getLogger(__name__)


def get_des_y3_coadd_tile_info(
        *, tilename, band, campaign, medsconf, piff_run):
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
    piff_run : str
        The PIFF PSF run to use.

    Returns
    -------
    info : dict
        A dictionary with at least the following keys:

            'wcs' : the coadd `esutil.wcsutil.WCS` object
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
            'image_flags' : any flags for the coadd image
            'scale' : a multiplicative factor to apply to the image
                (`*= scale`) and weight map (`/= scale**2`) for magnitude
                zero-point calibration.
            'magzp' : the magnitude zero point for the image

        The dictionaries in the 'src_info' list have at least the
        following keys:

            'scamp_wcs' : the SE `esutil.wcsutil.WCS` object with the
                scamp solution
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
            'psf_path' : the path to the PSFEx PSF model
            'piff_path' : the path to the Piff PSF model
            'piff_psf' : a piff.PSF object with the Piff PSF reconstruction.
            'scale' : a multiplicative factor to apply to the image
                (`*= scale`) and weight map (`/= scale**2`) for magnitude
                zero-point calibration
            'magzp' : the magnitude zero point for the image
            'image_flags' : any flags for the SE image
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

    info['image_flags'] = 0  # TODO set this properly for the coadd?

    for ii in info['src_info']:
        ii['image_flags'] = 0

        ii['image_ext'] = 'sci'

        ii['weight_path'] = ii['image_path']
        ii['weight_ext'] = 'wgt'

        ii['bmask_path'] = ii['image_path']
        ii['bmask_ext'] = 'msk'

        ii['bkg_ext'] = 'sci'

        # wcs info
        ii['scamp_wcs'] = eu.wcsutil.WCS(
            _munge_fits_header(
                fitsio.read_header(ii['image_path'], ext='sci')))
        ii['position_offset'] = POSITION_OFFSET

        # psfex psf
        ii['psfex_psf'] = galsim.des.DES_PSFEx(ii['psf_path'])

        # piff
        piff_data = load_piff_from_image_path(
            image_path=ii['image_path'],
            piff_run=piff_run,
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

            # HACK at the internals to code around a bug!
            if isinstance(ii['pixmappy_wcs'].origin, galsim._galsim.PositionD):
                logger.debug("adjusting the pixmappy origin to fix a bug!")
                ii['pixmappy_wcs']._origin = galsim.PositionD(
                    ii['pixmappy_wcs']._origin.x,
                    ii['pixmappy_wcs']._origin.y)

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
