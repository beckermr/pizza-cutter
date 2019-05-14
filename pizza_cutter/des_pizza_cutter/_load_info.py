import copy

import esutil as eu
import galsim.des
import galsim.config
import fitsio
import logging

import pixmappy

from ._piff_tools import get_piff_psf

logger = logging.getLogger(__name__)


def load_objects_into_info(*, info):
    """Load the data from objects in the info structure.

    NOTE: This function adds the following keys to `info` or the 'src_info'

        'image_wcs' - the WCS from the image header as an esutil.wcsutil.WCS
            object
        'psfex_psf' - the PSFEX psf (if 'psfex_path' is set) with the PSFEx
            PSF model
        'piff_psf' - the PIFF PSF model if the 'piff_path' is set
        'piff_wcs' - the WCS assumed by the Piff model if the 'piff_path'
            is set
        'pixmappy_wcs' - the pixmappy WCS attached to the PIFF model if the
            PIFF WCS is from pixmappy

    Parameters
    ----------
    info : dict
        A dictionary with at least the following keys:

            'src_info' : list of dicts for the SE sources
            'image_path' : the path to the FITS file with the coadd image
            'image_ext' : the name of the FITS extension with the coadd image

        The dictionaries in the 'src_info' list have at least the
        following keys:

            'image_path' : the path to the FITS file with the SE image
            'image_ext' : the name of the FITS extension with the SE image

        The source info structures can optionally have the keys

            'psfex_path' : the path to the PSFEx PSF model
            'piff_path' : the path to the Piff PSF model
            'galsim_psf_config' : a dictionary with a valid galsim config
                file entry to build the PSF as a galsim object.
    """
    try:
        info['image_wcs'] = eu.wcsutil.WCS(
            _munge_fits_header(fitsio.read_header(
                info['image_path'], ext=info['image_ext'])))
    except Exception:
        info['image_wcs'] = None

    # this is to keep track where it will be in image info extension
    info['file_id'] = 0

    for index, ii in enumerate(info['src_info']):
        # wcs info
        try:
            ii['image_wcs'] = eu.wcsutil.WCS(
                _munge_fits_header(
                    fitsio.read_header(ii['image_path'], ext=ii['image_ext'])))
        except Exception:
            ii['image_wcs'] = None

        # this is to keep track where it will be in image info extension
        ii['file_id'] = index+1

        # galsim objects
        if 'galsim_psf_config' in ii:
            ii['galsim_psf'] = _build_gsobject(ii['galsim_psf_config'])

        # psfex
        if 'psfex_path' in ii and ii['psfex_path'] is not None:
            try:
                ii['psfex_psf'] = galsim.des.DES_PSFEx(ii['psfex_path'])
            except Exception:
                logger.error(
                    'could not load PSFEx data at "%s"', ii['psfex_path'])
                ii['psfex_psf'] = None

        # piff
        if 'piff_path' in ii and ii['piff_path'] is not None:
            try:
                ii['piff_psf'] = get_piff_psf(ii['piff_path'])
                ii['piff_wcs'] = ii['piff_psf'].wcs[0]

                # try and grab pixmappy from piff
                if isinstance(ii['piff_psf'].wcs[0], pixmappy.GalSimWCS):
                    ii['pixmappy_wcs'] = ii['piff_psf'].wcs[0]

                    # HACK at the internals to code around a bug!
                    if isinstance(
                            ii['pixmappy_wcs'].origin,
                            galsim._galsim.PositionD):
                        logger.debug(
                            "adjusting the pixmappy origin to fix a bug!")
                        ii['pixmappy_wcs']._origin = galsim.PositionD(
                            ii['pixmappy_wcs']._origin.x,
                            ii['pixmappy_wcs']._origin.y)
                else:
                    ii['pixmappy_wcs'] = None
            except Exception:
                logger.error(
                    'could not load PIFF data at "%s"', ii['piff_path'])
                ii['piff_psf'] = None
                ii['piff_wcs'] = None
                ii['pixmappy_wcs'] = None

    return info


def _munge_fits_header(hdr):
    dct = {}
    for k in hdr.keys():
        try:
            dct[k.lower()] = hdr[k]
        except Exception:
            pass
    return dct


def _build_gsobject(config):
    dct = copy.deepcopy(config)
    _psf, safe = galsim.config.BuildGSObject({'blah': dct}, 'blah')
    assert safe, (
        "You must provide a reusable PSF object for galsim object PSFs")
    return _psf
