import copy
from os.path import expandvars
import esutil as eu
import galsim.des
import galsim.config
import fitsio
import logging

import pixmappy

from ._piff_tools import get_piff_psf
from ._affine_wcs import AffineWCS
from ..slice_utils.pbar import PBar

logger = logging.getLogger(__name__)


class _WCS(eu.wcsutil.WCS):
    """It turns out hashing this WCS class is really slow and done a lot.

    This version has a stable hash/repr computed once.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = super().__repr__()

    def __repr__(self):
        return self._state

    def __str__(self):
        return self._state

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return hash(self) == hash(other)


def load_objects_into_info(*, info, verbose=True, skip_se=False):
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

        The info structure can optionally have the key

            'affine_wcs_config' : a dictionary used to build an `AffineWCS`
                instance

        The source info structures can optionally have the keys

            'psfex_path' : the path to the PSFEx PSF model
            'piff_path' : the path to the Piff PSF model
            'galsim_psf_config' : a dictionary with a valid galsim config
                file entry to build the PSF as a galsim object.
            'affine_wcs_config' : a dictionary used to build an `AffineWCS`
                instance
    verbose : bool, optional
        If True, make noise, otherwise do not. Default is True.
    skip_se : bool, optional
        If True, do not load various PSFs and WCS solutions for each SE source.
        Default is False.
    """
    if verbose:
        print('loading coadd image data', flush=True)
    logger.info("loading image data products for %s/%s", info["path"], info["filename"])
    try:
        info['image_wcs'] = _WCS(
            _munge_fits_header(fitsio.read_header(
                expandvars(info['image_path']), ext=info['image_ext'])))
    except Exception as e:
        if verbose:
            print(
                "failed to load coadd image WCS do to error: %s" % repr(e),
                flush=True,
            )
        if "affine_wcs_config" not in info:
            raise e
        else:
            info['image_wcs'] = None

    if 'affine_wcs_config' in info:
        info['affine_wcs'] = AffineWCS(**info['affine_wcs_config'])

    # this is to keep track where it will be in image info extension
    info['file_id'] = 0

    # we need tuples for hashing
    if 'image_shape' in info:
        info['image_shape'] = tuple(info['image_shape'])

    if verbose:
        _itrbl = PBar(
            enumerate(info['src_info']),
            total=len(info["src_info"]),
            desc="loading SE image data",
        )
    else:
        _itrbl = enumerate(info['src_info'])
    for index, ii in _itrbl:
        # this is to keep track where it will be in image info extension
        ii['file_id'] = index+1

        # we need tuples for hashing
        if 'image_shape' in ii:
            ii['image_shape'] = tuple(ii['image_shape'])

        # we do these now for hashing and caching later
        # they are here for sims so NBD
        if 'affine_wcs_config' in ii:
            ii['affine_wcs'] = AffineWCS(**ii['affine_wcs_config'])

        if 'galsim_psf_config' in ii:
            ii['galsim_psf'] = _build_gsobject(ii['galsim_psf_config'])

        if not skip_se:
            logger.info(
                "loading image data products for %s/%s",
                ii["path"],
                ii["filename"],
            )

            # wcs info
            try:
                ii['image_wcs'] = _WCS(
                    _munge_fits_header(
                        fitsio.read_header(
                            expandvars(ii['image_path']), ext=ii['image_ext'])))
            except Exception as e:
                if verbose:
                    print(
                        "failed to load SE image WCS do to error: %s" % repr(e),
                        flush=True,
                    )
                if "affine_wcs_config" not in ii:
                    raise e
                else:
                    ii['image_wcs'] = None

            # psfex
            if 'psfex_path' in ii and ii['psfex_path'] is not None:
                psfex_path = expandvars(ii['psfex_path'])
                try:
                    ii['psfex_psf'] = galsim.des.DES_PSFEx(psfex_path)
                except Exception:
                    logger.error(
                        'could not load PSFEx data at "%s"', psfex_path)
                    ii['psfex_psf'] = None

            # piff
            if 'piff_path' in ii and ii['piff_path'] is not None:
                piff_path = expandvars(ii['piff_path'])
                try:
                    ii['piff_psf'] = get_piff_psf(piff_path)
                    ii['piff_wcs'] = ii['piff_psf'].wcs[0]

                    # try and grab pixmappy from piff
                    if isinstance(ii['piff_psf'].wcs[0], pixmappy.GalSimWCS):
                        ii['pixmappy_wcs'] = ii['piff_psf'].wcs[0]

                        # HACK at the internals to code around a bug!
                        if isinstance(
                                ii['pixmappy_wcs'].origin,
                                galsim._galsim.PositionD):
                            logger.warning(
                                "adjusting the pixmappy origin to fix a bug!"
                            )
                            ii['pixmappy_wcs']._origin = galsim.PositionD(
                                ii['pixmappy_wcs']._origin.x,
                                ii['pixmappy_wcs']._origin.y)
                    else:
                        ii['pixmappy_wcs'] = None
                except Exception:
                    logger.error(
                        'could not load PIFF data at "%s"', piff_path)
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
