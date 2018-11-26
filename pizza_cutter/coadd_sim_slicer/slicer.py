import os
import subprocess
import json
import gc

import numpy as np
import fitsio
import psfex
import meds
import esutil as eu
from meds.maker import MEDS_FMT_VERSION
from meds.util import (
    get_image_info_struct, get_meds_output_struct, validate_meds)
from esutil.wcsutil import WCS

from .._version import __version__
from ..files import StagedOutFile

# these are constants that are etched in stone for MEDS files
MAGZP_REF = 30.0
OBJECT_DATA_EXTNAME = 'object_data'
IMAGE_INFO_EXTNAME = 'image_info'
METADATA_EXTNAME = 'metadata'

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


def make_meds_pizza_slices(
        *,
        config,
        central_size, buffer_size,
        meds_path,
        image_path, image_ext,
        weight_path, weight_ext,
        bmask_path, bmask_ext,
        bkg_path=None, bkg_ext=None,
        seg_path=None, seg_ext=None,
        psf,
        fpack_pars=None,
        seed,
        tmpdir=None,
        remove_fits_file=True):
    """Build a MEDS pizza slices file.

    Parameters
    ----------
    config : str
        The input config file as a string.
    central_size : int
        Size of the central region for metadetection in pixels.
    buffer_size : int
        Size of the buffer around the central region in pixels.
    meds_path : str
        Path to the output MEDS file.
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
    noise_path : str
        Path to the noise image.
    noise_ext : int or str
        FITS extension for the noise image.
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
    fpack_pars : dict, optional
        A dictionary of fpack header keywords for compression.
    seed : int
        The random seed used to make the noise field.
    remove_fits_file : bool, optional
        If `True`, remove the FITS file after fpacking.
    tmpdir : str, optional
        A temporary directory to use. If `None`
    """

    metadata = _build_metadata(config)
    image_info = _build_image_info(
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

    with StagedOutFile(meds_path + '.fz', tmpdir) as sf:
        tmp_meds_file = sf.path.replace('.fz', '')
        with fitsio.FITS(tmp_meds_file, 'rw', clobber=True) as fits:
            # make the image data here since we need to have the file open to
            # fill the arrays on disk
            _slice_coadd_image(
                central_size=central_size,
                buffer_size=buffer_size,
                image_path=image_path,
                image_ext=image_ext,
                bkg_path=bkg_path,
                bkg_ext=bkg_ext,
                weight_path=weight_path,
                weight_ext=weight_ext,
                seg_path=seg_path,
                seg_ext=seg_ext,
                bmask_path=bmask_path,
                bmask_ext=bmask_ext,
                psf=psf,
                fits=fits,
                fpack_pars=fpack_pars,
                seed=seed)

            fits.write(image_info, extname=IMAGE_INFO_EXTNAME)
            fits.write(metadata, extname=METADATA_EXTNAME)

        # fpack it
        cmd = 'fpack %s' % tmp_meds_file
        print("fpacking:\n    command: '%s'" % cmd, flush=True)
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception:
            pass
        else:
            if remove_fits_file:
                os.remove(tmp_meds_file)

        # validate the fpacked file
        print('validating:', flush=True)
        validate_meds(sf.path)


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


def _build_metadata(config):
    """Build the metadata for the pizza slices MEDS file"""
    numpy_version = np.__version__
    esutil_version = eu.__version__
    fitsio_version = fitsio.__version__
    meds_version = meds.__version__
    dt = [
        ('magzp_ref', 'f8'),
        ('config', 'S%d' % len(config)),
        ('pizza_cutter_version', 'S%d' % len(__version__)),
        ('numpy_version', 'S%d' % len(numpy_version)),
        ('esutil_version', 'S%d' % len(esutil_version)),
        ('fitsio_version', 'S%d' % len(fitsio_version)),
        ('meds_version', 'S%d' % len(meds_version)),
        ('meds_fmt_version', 'S%d' % len(MEDS_FMT_VERSION))]
    metadata = np.zeros(1, dt)
    metadata['magzp_ref'] = MAGZP_REF
    metadata['config'] = config
    metadata['numpy_version'] = numpy_version
    metadata['esutil_version'] = esutil_version
    metadata['fitsio_version'] = fitsio_version
    metadata['meds_version'] = meds_version
    metadata['meds_fmt_version'] = MEDS_FMT_VERSION
    metadata['pizza_cutter_version'] = __version__
    return metadata


def _slice_coadd_image(
        *,
        central_size, buffer_size,
        image_path, image_ext,
        bkg_path, bkg_ext,
        weight_path, weight_ext,
        seg_path, seg_ext,
        bmask_path, bmask_ext,
        psf,
        fits,
        fpack_pars,
        seed):
    """Slice a coadd image into metadetection regions, making a MEDS file."""

    # make object data so we can write the cutouts
    wcs = _read_data(
        image_path=image_path,
        image_ext=image_ext,
        bkg_path=bkg_path,
        bkg_ext=bkg_ext,
        weight_path=weight_path,
        weight_ext=weight_ext,
        seg_path=seg_path,
        seg_ext=seg_ext,
        bmask_path=bmask_path,
        bmask_ext=bmask_ext,
        psf=psf,
        which='wcs')
    object_data = _build_object_data(
        central_size=central_size,
        buffer_size=buffer_size,
        image_width=wcs.get_naxis()[0],
        wcs=wcs)

    pex = _read_data(
        image_path=image_path,
        image_ext=image_ext,
        bkg_path=bkg_path,
        bkg_ext=bkg_ext,
        weight_path=weight_path,
        weight_ext=weight_ext,
        seg_path=seg_path,
        seg_ext=seg_ext,
        bmask_path=bmask_path,
        bmask_ext=bmask_ext,
        psf=psf,
        which='psf')
    _fill_psf_data_and_write_psf_cutouts(
        pex=pex,
        object_data=object_data,
        fits=fits,
        fpack_pars=fpack_pars)
    fits.write(object_data, extname=OBJECT_DATA_EXTNAME)

    # now go one by one and write the different images
    img = _read_data(
        image_path=image_path,
        image_ext=image_ext,
        bkg_path=bkg_path,
        bkg_ext=bkg_ext,
        weight_path=weight_path,
        weight_ext=weight_ext,
        seg_path=seg_path,
        seg_ext=seg_ext,
        bmask_path=bmask_path,
        bmask_ext=bmask_ext,
        psf=psf,
        which='img')
    if bkg_path is not None and bkg_ext is not None:
        img -= _read_data(
            image_path=image_path,
            image_ext=image_ext,
            bkg_path=bkg_path,
            bkg_ext=bkg_ext,
            weight_path=weight_path,
            weight_ext=weight_ext,
            seg_path=seg_path,
            seg_ext=seg_ext,
            bmask_path=bmask_path,
            bmask_ext=bmask_ext,
            psf=psf,
            which='bkg')
    _write_cutouts(
        im=img,
        ext=IMAGE_CUTOUT_EXTNAME,
        object_data=object_data,
        fits=fits,
        fpack_pars=fpack_pars)
    del img
    gc.collect()

    wgt = _read_data(
        image_path=image_path,
        image_ext=image_ext,
        bkg_path=bkg_path,
        bkg_ext=bkg_ext,
        weight_path=weight_path,
        weight_ext=weight_ext,
        seg_path=seg_path,
        seg_ext=seg_ext,
        bmask_path=bmask_path,
        bmask_ext=bmask_ext,
        psf=psf,
        which='wgt')
    _write_cutouts(
        im=wgt,
        ext=WEIGHT_CUTOUT_EXTNAME,
        object_data=object_data,
        fits=fits,
        fpack_pars=fpack_pars)

    rng = np.random.RandomState(seed=seed)
    noise = rng.normal(size=wcs.get_naxis()) * np.sqrt(1.0 / wgt)
    _write_cutouts(
        im=noise,
        ext=NOISE_CUTOUT_EXTNAME,
        object_data=object_data,
        fits=fits,
        fpack_pars=fpack_pars)
    del noise
    del wgt
    gc.collect()

    if seg_path is not None and seg_ext is not None:
        seg = _read_data(
            image_path=image_path,
            image_ext=image_ext,
            bkg_path=bkg_path,
            bkg_ext=bkg_ext,
            weight_path=weight_path,
            weight_ext=weight_ext,
            seg_path=seg_path,
            seg_ext=seg_ext,
            bmask_path=bmask_path,
            bmask_ext=bmask_ext,
            psf=psf,
            which='seg')
        _write_cutouts(
            im=seg,
            ext=SEG_CUTOUT_EXTNAME,
            object_data=object_data,
            fits=fits,
            fpack_pars=fpack_pars)
        del seg
        gc.collect()

    msk = _read_data(
        image_path=image_path,
        image_ext=image_ext,
        bkg_path=bkg_path,
        bkg_ext=bkg_ext,
        weight_path=weight_path,
        weight_ext=weight_ext,
        seg_path=seg_path,
        seg_ext=seg_ext,
        bmask_path=bmask_path,
        bmask_ext=bmask_ext,
        psf=psf,
        which='msk')
    _write_cutouts(
        im=msk,
        ext=BMASK_CUTOUT_EXTNAME,
        object_data=object_data,
        fits=fits,
        fpack_pars=fpack_pars)
    del msk
    gc.collect()


def _fill_psf_data_and_write_psf_cutouts(
        *,
        pex,
        object_data,
        fits,
        fpack_pars):
    """fills the PSF data and writes the cutouts"""

    psf_start_row = 0
    psf_shape = None
    for iobj in range(len(object_data)):
        row = object_data['orig_row'][iobj, 0]
        col = object_data['orig_col'][iobj, 0]

        pim = pex.get_rec(row, col)
        cen = pex.get_center(row, col)
        try:
            sigma = pex.get_sigma(row, col)
        except Exception:
            sigma = pex.get_sigma()

        if psf_shape is None:
            psf_shape = pim.shape
            psf_npix = psf_shape[0]**2
            # set for all objects here
            object_data['psf_box_size'] = psf_shape[0]

            # reserve the mosaic here
            print(
                '%s:\n    reserving mosaic and '
                'writing data' % PSF_CUTOUT_EXTNAME,
                flush=True)
            fits.create_image_hdu(
                img=None,
                dtype=CUTOUT_DTYPES[PSF_CUTOUT_EXTNAME],
                dims=[len(object_data) * psf_npix],
                extname=PSF_CUTOUT_EXTNAME,
                header=fpack_pars)

            # now need to write the header
            fits[PSF_CUTOUT_EXTNAME].write_keys(fpack_pars, clean=False)
        else:
            tpsf_shape = pim.shape
            if tpsf_shape != psf_shape:
                raise ValueError("currently all psfs "
                                 "must be same size")

        object_data['psf_cutout_row'][iobj, 0] = cen[0]
        object_data['psf_cutout_col'][iobj, 0] = cen[1]
        object_data['psf_sigma'][iobj, 0] = sigma
        object_data['psf_start_row'][iobj, 0] = psf_start_row

        fits[PSF_CUTOUT_EXTNAME].write(pim, start=psf_start_row)

        psf_start_row += psf_npix


def _write_cutouts(*, im, ext, object_data, fits, fpack_pars):
    """Write the cutouts of a given image to an ext"""

    print('%s:\n    reserving mosaic' % ext, flush=True)

    dims = [int(np.sum(object_data['box_size'] * object_data['box_size']))]

    # this reserves space for the images and header,
    # but no data is written
    fits.create_image_hdu(
        img=None,
        dtype=CUTOUT_DTYPES[ext],
        dims=dims,
        extname=ext,
        header=fpack_pars)

    # now need to write the header
    fits[ext].write_keys(fpack_pars, clean=False)

    # finally write the data to the image
    print('    writing mosaic', flush=True)
    for i in range(len(object_data)):
        orow = object_data['orig_start_row'][i, 0]
        ocol = object_data['orig_start_col'][i, 0]
        bsize = object_data['box_size'][i]
        start_row = object_data['start_row'][i, 0]

        # nothing ever hits the edge here, but doing this anyways
        read_im = im[orow:orow + bsize, ocol:ocol+bsize]
        subim = np.zeros((bsize, bsize), dtype=CUTOUT_DTYPES[ext])
        subim += CUTOUT_DEFAULT_VALUES[ext]
        subim[:, :] = read_im

        fits[ext].write(subim, start=start_row)


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


def _read_data(
        *,
        image_path, image_ext,
        bkg_path, bkg_ext,
        weight_path, weight_ext,
        seg_path, seg_ext,
        bmask_path, bmask_ext,
        psf,
        which):
    """Read the data.

    Returns
    -------
    the requested item
    """

    if which == 'img':
        return fitsio.read(image_path, ext=image_ext)
    elif which == 'wcs':
        imh = fitsio.read_header(image_path)
        wcs_dict = {k.lower(): imh[k] for k in imh.keys()}
        return WCS(wcs_dict)
    elif which == 'wgt':
        return fitsio.read(weight_path, ext=weight_ext)
    elif which == 'bkg':
        return fitsio.read(bkg_path, ext=bkg_ext)
    elif which == 'seg':
        return fitsio.read(seg_path, ext=seg_ext)
    elif which == 'msk':
        return fitsio.read(bmask_path, ext=bmask_ext)
    elif which == 'psf':
        return psfex.PSFEx(psf)
    else:
        raise ValueError("Item '%s' not recognized!" % which)
