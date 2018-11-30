import numpy as np

import fitsio
from ngmix.medsreaders import NGMixMEDS
import esutil as eu

from .slicer import (
    _build_object_data,
    _build_image_info,
    _parse_psf,
    IMAGE_CUTOUT_EXTNAME,
    WEIGHT_CUTOUT_EXTNAME,
    SEG_CUTOUT_EXTNAME,
    BMASK_CUTOUT_EXTNAME,
    NOISE_CUTOUT_EXTNAME,
    CUTOUT_DTYPES,
    CUTOUT_DEFAULT_VALUES,
    MAGZP_REF)
from .memmappednoise import MemMappedNoiseImage


class CoaddSimSliceMEDS(NGMixMEDS):
    """A MEDS interface for slices of coadd simulations.

    Parameters
    ----------
    central_size : int
        Size of the central region for metadetection in pixels.
    buffer_size : int
        Size of the buffer around the central region in pixels.
    image_path : str
        Path to the coadd image.
    image_ext : int or str
        FITS extension for the coadd image.
    weight_path : str
        Path to the weight map.
    weight_ext : int or str
        FITS extension for the weight map.
    bmask_path : str
        Path to the bit mask.
    bmask_ext : int or str
        FITS extension for the bit mask.
    bkg_path : str, optional
        Path to the background image.
    bkg_ext : int or str, optional
        FITS extension of the background image.
    seg_path : str, optional
        Path to the seg map.
    seg_ext : int or str, optional
        FITS extension for the seg map.
    psf : str
        The path to the input PSFEX file.
    seed : int
        The random seed used to make the noise field.
    noise_size : int, optional
        The size of patches for generating the noise image.
    """
    def __init__(
            self, *, central_size, buffer_size, image_path, image_ext,
            bkg_path=None, bkg_ext=None, seg_path=None, seg_ext=None,
            weight_path, weight_ext, bmask_path, bmask_ext, psf, seed,
            noise_size=1000):

        # we need to set the slice properties here
        # they get used later to subset the images
        imh = fitsio.read_header(image_path)
        wcs_dict = {k.lower(): imh[k] for k in imh.keys()}
        wcs = eu.wcsutil.WCS(wcs_dict)
        self._cat = _build_object_data(
            central_size=central_size,
            buffer_size=buffer_size,
            image_width=wcs.get_naxis()[0],
            wcs=wcs)
        self._wcs = wcs

        # set the image info for compat with MEDS interface
        self._image_info = _build_image_info(
            image_path=image_path,
            image_ext=image_ext,
            bkg_path=bkg_path,
            bkg_ext=bkg_ext,
            weight_path=weight_path,
            weight_ext=weight_ext,
            seg_path=seg_path,
            seg_ext=seg_ext,
            bmask_path=bmask_path,
            bmask_ext=bmask_ext)

        # fill the PSF properties now
        self._pex = _parse_psf(psf=psf, wcs_dict=imh)
        for iobj in range(len(self._cat)):
            row = self._cat['orig_row'][iobj, 0]
            col = self._cat['orig_col'][iobj, 0]

            cen = self._pex.get_center(row, col)
            try:
                sigma = self._pex.get_sigma(row, col)
            except Exception:
                sigma = self._pex.get_sigma()

            self._cat['psf_cutout_row'][iobj, 0] = cen[0]
            self._cat['psf_cutout_col'][iobj, 0] = cen[1]
            self._cat['psf_sigma'][iobj, 0] = sigma

        # now finally we load fitsio interfaces for the images
        self._fits_objs = {}
        self._fits_objs['image'] = fitsio.FITS(
            image_path)
        self._image_ext = image_ext
        self._fits_objs['weight'] = fitsio.FITS(
            weight_path)
        self._weight_ext = weight_ext
        self._fits_objs['bmask'] = fitsio.FITS(
            bmask_path)
        self._bmask_ext = bmask_ext
        if bkg_path is not None:
            self._fits_objs['bkg'] = fitsio.FITS(
                bkg_path)
            self._bkg_ext = bkg_ext
        if seg_path is not None:
            self._fits_objs['seg'] = fitsio.FITS(
                seg_path)
            self._seg_ext = seg_ext
        self._noise_obj = MemMappedNoiseImage(
            seed=seed,
            weight=self._fits_objs['weight'][self._weight_ext],
            sx=noise_size, sy=noise_size)

        # set metadata just in case
        self._meta = np.zeros(1, dtype=[('magzp_ref', 'f8')])
        self._meta['magzp_ref'] = MAGZP_REF

    def close(self):
        """Close all of the FITS objects.
        """
        for _, f in self._fits_objs.items():
            f.close()

    def get_psf(self, iobj, icut):
        """Get a PSF image.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.

        Returns
        -------
        pim : np.array
            An image of the PSF.
        """
        row = self._cat['orig_row'][iobj, icut]
        col = self._cat['orig_col'][iobj, icut]
        return self._pex.get_rec(row, col)

    def get_cutout(self, iobj, icutout, type='image'):
        """Get a single cutout for the indicated entry and image type.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.
        type: string, optional
            Cutout type. Default is 'image'. Allowed values are 'image',
            'weight', 'seg', 'bmask', 'ormask', 'noise' or 'psf'.

        Returns
        -------
        cutout : np.array
            The cutout image.
        """

        if type == 'psf':
            return self.get_psf(iobj, icutout)

        self._check_indices(iobj, icutout=icutout)

        im, defval, dtype = self._get_image_and_defaults(type)
        orow = self._cat['orig_start_row'][iobj, icutout]
        ocol = self._cat['orig_start_col'][iobj, icutout]
        bsize = self._cat['box_size'][iobj]

        subim = np.zeros((bsize, bsize), dtype=dtype)
        subim += defval

        if im is not None:
            read_im = im[orow:orow+bsize, ocol:ocol+bsize].copy()
            if type == 'image' and 'bkg' in self._fits_objs:
                bkg, _, _ = self._get_image_and_defaults('bkg')
                read_im -= bkg[orow:orow+bsize, ocol:ocol+bsize]

            subim[:, :] = read_im

        return subim

    def _get_image_and_defaults(self, type):
        if type == 'image':
            return (
                self._fits_objs['image'][self._image_ext],
                CUTOUT_DEFAULT_VALUES[IMAGE_CUTOUT_EXTNAME],
                CUTOUT_DTYPES[IMAGE_CUTOUT_EXTNAME])
        elif type == 'weight':
            return (
                self._fits_objs['weight'][self._weight_ext],
                CUTOUT_DEFAULT_VALUES[WEIGHT_CUTOUT_EXTNAME],
                CUTOUT_DTYPES[WEIGHT_CUTOUT_EXTNAME])
        elif type == 'seg' and 'seg' in self._fits_objs:
            return (
                self._fits_objs['seg'][self._seg_ext],
                CUTOUT_DEFAULT_VALUES[SEG_CUTOUT_EXTNAME],
                CUTOUT_DTYPES[SEG_CUTOUT_EXTNAME])
        elif type == 'bmask':
            return (
                self._fits_objs['bmask'][self._bmask_ext],
                CUTOUT_DEFAULT_VALUES[BMASK_CUTOUT_EXTNAME],
                CUTOUT_DTYPES[BMASK_CUTOUT_EXTNAME])
        elif type == 'bkg' and 'bkg' in self._fits_objs:
            return (
                self._fits_objs['bkg'][self._bkg_ext],
                CUTOUT_DEFAULT_VALUES[IMAGE_CUTOUT_EXTNAME],
                CUTOUT_DTYPES[IMAGE_CUTOUT_EXTNAME])
        elif type == 'noise':
            return (
                self._noise_obj,
                CUTOUT_DEFAULT_VALUES[NOISE_CUTOUT_EXTNAME],
                CUTOUT_DTYPES[NOISE_CUTOUT_EXTNAME])
        else:
            raise ValueError("Image type '%s' not recognized!" % type)
