import json
import copy

import numpy as np
import galsim
import fitsio
from ngmix.medsreaders import NGMixMEDS
import esutil as eu
from meds.util import get_image_info_struct, get_meds_output_struct

from .memmappednoise import MemMappedNoiseImage
from ..simulation.maskgen import BMaskGenerator
from .galsim_psf import GalSimPSF, GalSimPSFEx

MAGZP_REF = 30.0
OBJECT_DATA_EXTNAME = 'object_data'
IMAGE_INFO_EXTNAME = 'image_info'

IMAGE_CUTOUT_EXTNAME = 'image_cutouts'
WEIGHT_CUTOUT_EXTNAME = 'weight_cutouts'
SEG_CUTOUT_EXTNAME = 'seg_cutouts'
BMASK_CUTOUT_EXTNAME = 'bmask_cutouts'
ORMASK_CUTOUT_EXTNAME = 'ormask_cutouts'
NOISE_CUTOUT_EXTNAME = 'noise_cutouts'
PSF_CUTOUT_EXTNAME = 'psf'
CUTOUT_DTYPES = {
    'image_cutouts': 'f4',
    'weight_cutouts': 'f4',
    'seg_cutouts': 'i4',
    'bmask_cutouts': 'i4',
    'ormask_cutouts': 'i4',
    'noise_cutouts': 'f4',
    'psf': 'f4'}
CUTOUT_DEFAULT_VALUES = {
    'image_cutouts': 0.0,
    'weight_cutouts': 0.0,
    'seg_cutouts': 0,
    'bmask_cutouts': 2**30,
    'ormask_cutouts': 2**30,
    'noise_cutouts': 0.0,
    'psf': 0.0}

# this is always true for these sims
POSITION_OFFSET = 1


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
    psf : str or dict
        The path to the input PSFEX file or a dictionary specifying the
        GalSim object to draw as the PSF.
    seed : int
        The random seed used to make the noise field.
    noise_size : int, optional
        The size of patches for generating the noise image.
    bmask_catalog : str, optional
        The path to the FITS file with the bit mask catalog for applying
        random bit masks. If `None`, the bit mask from the MEDS file is used
        instead.
    """
    def __init__(
            self, *, central_size, buffer_size, image_path, image_ext,
            bkg_path=None, bkg_ext=None, seg_path=None, seg_ext=None,
            weight_path, weight_ext, bmask_path, bmask_ext, psf, seed,
            noise_size=1000, bmask_catalog=None):

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
        self._wcs_dict = wcs_dict

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
        eval_locals = {
            'row_cen': (wcs.get_naxis()[0] - 1) / 2,
            'col_cen': (wcs.get_naxis()[1] - 1) / 2,
            'nrows': wcs.get_naxis()[0],
            'ncols': wcs.get_naxis()[1]}
        self._pex = _parse_psf(psf=psf, wcs_dict=imh, eval_locals=eval_locals)
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

        # deal with masking
        self._bmask_catalog = bmask_catalog
        if self._bmask_catalog is not None:
            self._bmask_gen = BMaskGenerator(
                bmask_cat=self._bmask_catalog, seed=seed+2)

    def get_psf_rec_func(self, iobj, icutout):
        """Get a function that returns the PSF image at a given row, col
        in each cutout.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.

        Returns
        -------
        func : function
            A function with call signature `func(row, col)` that returns an
            image of the PSF at the given location within the cutout.
        """

        row_start = copy.copy(self._cat['orig_start_row'][iobj, icutout])
        col_start = copy.copy(self._cat['orig_start_col'][iobj, icutout])
        try:
            _pex = self._pex.copy()
        except Exception:
            _pex = copy.deepcopy(self._pex)

        def _func(row, col):
            return _pex.get_rec(row + row_start, col + col_start)

        return _func

    def get_wcs_jacobian_func(self, iobj, icutout):
        """Get a function that returns the WCS Jacobian as a dictionary
        at each point in a cutout.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.

        Returns
        -------
        func : function
            A function with call signature `func(row, col)` that returns a
            dictionary containing the WCS Jacobian at the given input location.
        """

        row_start = copy.copy(self._cat['orig_start_row'][iobj, icutout])
        col_start = copy.copy(self._cat['orig_start_col'][iobj, icutout])

        wcs = eu.wcsutil.WCS(self._wcs_dict)

        def _func(row, col):
            jacob = wcs.get_jacobian(
                col + col_start + POSITION_OFFSET,
                row + row_start + POSITION_OFFSET)
            out = {}
            out['dudcol'] = jacob[0]
            out['dudrow'] = jacob[1]
            out['dvdcol'] = jacob[2]
            out['dvdrow'] = jacob[3]
            return out

        return _func

    def close(self):
        """Close all of the FITS objects.
        """
        for _, f in self._fits_objs.items():
            f.close()
        if hasattr(self, '_bmask_gen'):
            self._bmask_gen.close()

    def get_psf(self, iobj, icutout):
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
        row = self._cat['orig_row'][iobj, icutout]
        col = self._cat['orig_col'][iobj, icutout]
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
            'weight', 'seg', 'bmask', 'ormask', 'noise', or 'psf'.

        Returns
        -------
        cutout : np.array
            The cutout image.
        """

        if type == 'psf':
            return self.get_psf(iobj, icutout)

        if type == 'bmask' and self._bmask_catalog is not None:
            # this approximately puts the masks in the right coords for the
            # coadd
            return self._bmask_gen.get_bmask(iobj)[::-1, ::-1].T

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
        elif type == 'noise_for_noise_interp':
            return (
                self._noise_for_noise_interp_obj,
                CUTOUT_DEFAULT_VALUES[NOISE_CUTOUT_EXTNAME],
                CUTOUT_DTYPES[NOISE_CUTOUT_EXTNAME])
        else:
            raise ValueError("Image type '%s' not recognized!" % type)


def _build_image_info(
        *,
        image_path, image_ext,
        bkg_path, bkg_ext,
        weight_path, weight_ext,
        seg_path, seg_ext,
        bmask_path, bmask_ext):
    """Build the image info structure."""

    imh = fitsio.read_header(image_path)
    wcs_dict = {k.lower(): imh[k] for k in imh.keys()}
    wcs_json = json.dumps(wcs_dict)
    strlen = np.max([len(image_path), len(weight_path), len(bmask_path)])
    if seg_path is not None:
        strlen = max([strlen, len(seg_path)])
    if bkg_path is not None:
        strlen = max([strlen, len(bkg_path)])
    ii = get_image_info_struct(
        1,
        strlen,
        wcs_len=len(wcs_json))
    ii['image_path'] = image_path
    ii['image_ext'] = image_ext
    ii['weight_path'] = weight_path
    ii['weight_ext'] = weight_ext
    if seg_path is not None:
        ii['seg_path'] = seg_path
        ii['seg_ext'] = seg_ext
    ii['bmask_path'] = bmask_path
    ii['bmask_ext'] = bmask_ext
    if bkg_path is not None:
        ii['bkg_path'] = bkg_path
        ii['bkg_ext'] = bkg_ext
    ii['image_id'] = 0
    ii['image_flags'] = 0
    ii['magzp'] = 30.0
    ii['scale'] = 10.0**(0.4*(MAGZP_REF - ii['magzp']))
    ii['position_offset'] = POSITION_OFFSET
    ii['wcs'] = wcs_json

    return ii


def _build_object_data(
        *,
        central_size,
        buffer_size,
        image_width,
        wcs):
    """Build the internal MEDS data structure.

    Information about the PSF is filled when the PSF is drawn
    """
    # compute the number of pizza slices
    # - each slice is central_size + 2 * buffer_size wide
    #   because a buffer is on each side
    # - each pizza slice is moved over by central_size to tile the full
    #  coadd tile
    # - we leave a buffer on either side so that we have to tile only
    #  im_width - 2 * buffer_size pixels
    box_size = central_size + 2 * buffer_size
    cen_offset = (central_size - 1) / 2.0  # zero-indexed at pixel centers

    nobj = (image_width - 2 * buffer_size) / central_size
    if int(nobj) != nobj:
        raise ValueError(
            "The coadd tile must be exactly tiled by the pizza slices!")
    nobj = int(nobj)

    # now make the object data and extra fields for the PSF
    # we use an namx of 2 because if we set 1, then the fields come out with
    # the wrong shape when we read them from fitsio
    nmax = 2
    psf_dtype = [
        ('psf_box_size', 'i4'),
        ('psf_cutout_row', 'f8', nmax),
        ('psf_cutout_col', 'f8', nmax),
        ('psf_sigma', 'f4', nmax),
        ('psf_start_row', 'i8', nmax)]
    output_info = get_meds_output_struct(
        nobj * nobj, nmax, extra_fields=psf_dtype)

    # and fill!
    output_info['id'] = np.arange(nobj * nobj)
    output_info['box_size'] = box_size
    output_info['ncutout'] = 1
    output_info['file_id'] = 0

    start_row = 0
    box_size2 = box_size * box_size
    for icol in range(nobj):
        for irow in range(nobj):
            index = icol * nobj + irow

            # center of the cutout
            col_or_x = buffer_size + icol * central_size + cen_offset
            row_or_y = buffer_size + irow * central_size + cen_offset

            wcs_col = col_or_x + POSITION_OFFSET
            wcs_row = row_or_y + POSITION_OFFSET

            ra, dec = wcs.image2sky(wcs_col, wcs_row)
            jacob = wcs.get_jacobian(wcs_col, wcs_row)

            output_info['ra'][index] = ra
            output_info['dec'][index] = dec
            output_info['start_row'][index, 0] = start_row
            output_info['orig_row'][index, 0] = row_or_y
            output_info['orig_col'][index, 0] = col_or_x
            output_info['orig_start_row'][index, 0] = (
                row_or_y - cen_offset - buffer_size)
            output_info['orig_start_col'][index, 0] = (
                col_or_x - cen_offset - buffer_size)
            output_info['cutout_row'][index, 0] = cen_offset
            output_info['cutout_col'][index, 0] = cen_offset
            output_info['dudcol'][index, 0] = jacob[0]
            output_info['dudrow'][index, 0] = jacob[1]
            output_info['dvdcol'][index, 0] = jacob[2]
            output_info['dvdrow'][index, 0] = jacob[3]

            start_row += box_size2

    return output_info


def _parse_psf(*, psf, wcs_dict, eval_locals=None):
    if isinstance(psf, dict):
        return GalSimPSF(
            psf,
            wcs=galsim.FitsWCS(header=wcs_dict),
            eval_locals=eval_locals)
    else:
        return GalSimPSFEx(psf)
