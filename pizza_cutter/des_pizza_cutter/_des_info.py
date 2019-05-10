import logging

import desmeds

from ._constants import MAGZP_REF, POSITION_OFFSET
from ._piff_tools import load_piff_path_from_image_path

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
            'psfex_path' : the path to the PSFEx PSF model
            'piff_path' : the path to the Piff PSF model
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
    info['tilename'] = tilename
    info['band'] = band
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

    # this is to keep track where it will be in image info extension
    info['image_id'] = 0

    for index, ii in enumerate(info['src_info']):
        # this is to keep track where it will be in image info extension
        ii['image_id'] = index+1

        ii['image_flags'] = 0

        ii['image_ext'] = 'sci'

        ii['weight_path'] = ii['image_path']
        ii['weight_ext'] = 'wgt'

        ii['bmask_path'] = ii['image_path']
        ii['bmask_ext'] = 'msk'

        ii['bkg_ext'] = 'sci'

        # wcs info
        ii['position_offset'] = POSITION_OFFSET

        # psfex psf
        ii['psfex_path'] = ii['psf_path']

        # piff
        if piff_run is not None:
            piff_data = load_piff_path_from_image_path(
                image_path=ii['image_path'],
                piff_run=piff_run,
            )

            ii['piff_path'] = piff_data['psf_path']
            ii['image_flags'] |= piff_data['flags']

        # image scale
        ii['scale'] = 10.0**(0.4*(MAGZP_REF - ii['magzp']))

    return info
