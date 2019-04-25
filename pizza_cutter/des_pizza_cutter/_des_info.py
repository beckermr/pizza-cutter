import galsim
import esutil as eu
import fitsio

from meds.bounds import Bounds
import psfex

from ._sky_bounds import get_rough_sky_bounds
from ._constants import MAGZP_REF, POSITION_OFFSET


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
        The MEDS version. This string is used to set the download directory
        for the files for subsequent downloads.

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
            'position_offset' : the offset to add to zero-indexed image
                coordinates to get transform them to the convention assumed
                by the WCS.
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
            'psf_rec' : an object with the PSF reconstruction. This object
                will have the methods `get_rec(row, col)` and
                `get_center(row, col)` for getting an image of the PSF and
                its center.
            'sky_bnds' : a `meds.meds.Bounds` object with the bounds of the
                SE image in a (u, v) spherical coordinate system about the
                center. See the documentation of `get_rough_sky_bounds` for
                more details on how to use this object.
            'ra_ccd' : the RA of the SE image center in decimal degreees
            'dec_ccd' : the DEC of the SE image center in decimal degrees
            'ccd_bnds' : a `meds.meds.Bounds` object in zero-indexed image
                coordinates
            'scale' : a multiplicative factor to apply to the image
                (`*= scale`) and weight map (`/= scale**2`) for magnitude
                zero-point calibration.
    coadd : `DESCoadd`
        The `DESCoadd` object that can be used to download the data via the
        `download()` method.
    """

    # guarding this here, since not all codes would need it
    import desmeds

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

    for ii in info['src_info']:
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

        # psf
        ii['psf_rec'] = psfex.PSFEx(ii['psf_path'])

        # rough sky cut tests
        ncol, nrow = ii['wcs'].get_naxis()
        sky_bnds, ra_ccd, dec_ccd = get_rough_sky_bounds(
            wcs=ii['wcs'],
            position_offset=POSITION_OFFSET,
            bounds_buffer_uv=16.0,
            n_grid=4)
        ii['sky_bnds'] = sky_bnds
        ii['ra_ccd'] = ra_ccd
        ii['dec_ccd'] = dec_ccd
        ii['ccd_bnds'] = Bounds(0, nrow-1, 0, ncol-1)
        ii['scale'] = 10.0**(0.4*(MAGZP_REF - ii['magzp']))

    return info, coadd


def _munge_fits_header(hdr):
    dct = {}
    for k in hdr.keys():
        try:
            dct[k.lower()] = hdr[k]
        except Exception:
            pass
    return dct
