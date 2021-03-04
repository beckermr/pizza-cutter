import os
from os.path import expandvars
import subprocess
import functools
import json
import logging

import numpy as np
import fitsio
import meds
import esutil as eu
import piff
import pixmappy
import desmeds

from meds.maker import MEDS_FMT_VERSION
from meds.util import (
    get_image_info_struct, get_meds_output_struct, validate_meds)

from .._version import __version__
from ._constants import (
    METADATA_EXTNAME,
    MAGZP_REF,
    IMAGE_INFO_EXTNAME,
    OBJECT_DATA_EXTNAME,
    IMAGE_CUTOUT_EXTNAME,
    WEIGHT_CUTOUT_EXTNAME,
    SEG_CUTOUT_EXTNAME,
    BMASK_CUTOUT_EXTNAME,
    ORMASK_CUTOUT_EXTNAME,
    NOISE_CUTOUT_EXTNAME,
    PSF_CUTOUT_EXTNAME,
    EPOCHS_INFO_EXTNAME,
    CUTOUT_DTYPES,
    CUTOUT_DEFAULT_VALUES)
from ..slice_utils.locate import build_slice_locations
from ..slice_utils.measure import measure_fwhm
from ..files import StagedOutFile
from ._coadd_slices import (
    _build_slice_inputs, _coadd_slice_inputs)

from ..slice_utils.pbar import prange

logger = logging.getLogger(__name__)


def make_des_pizza_slices(
        *,
        config,
        meds_path,
        info,
        seed,
        remove_fits_file=True,
        tmpdir=None,
        fpack_pars=None,
        coadd_config,
        single_epoch_config):
    """Build a MEDS pizza slices file.

    Parameters
    ----------
    config : str
        The input config file as a string.
    meds_path : str
        Path to the output MEDS file.
    info : dict
        Dictionary of information about the coadd and SE images. This should
        be set to the output of the `des-pizza-cutter-prep-tile` CLI or a
        similar function.
    seed : int
        The random seed used to make the noise field.
    remove_fits_file : bool, optional
        If `True`, remove the FITS file after fpacking. Only works if not
        using a temporary directory.
    tmpdir : optional, string
        Optional temporary directory to use for staging files
    fpack_pars : dict, optional
        A dictionary of fpack header keywords for compression.
    coadd_config : dict
        A dictionary with the configuration parameters for the coadd image
        slices and weighting. Thew required entries are

        central_size : int
            Size of the central region for metadetection in pixels.
        buffer_size : int
            Size of the buffer around the central region in pixels.
        coadding_weight : str
            The kind of relative weight to apply to each of the SE images that
            form a coadd. The options are

            'noise' - use the maximum of the weight map for each SE image.
            'noise-fwhm' - use the maximum of the weight map divided by the
                (PSF FWHM)**4
        psf_box_size : int
            The size of the PSF stamp in the final coadd coordinates. This
            should be an odd number large enough to contain any SE PSF.

    single_epoch_config : dict
        This is a dictionary with the configuration options for processing
        the single epoch images. See the documentaion of
        `pizza_cutter.des_pizza_cutter._coadd_slices._build_slice_inputs`
        for details on the required entries.
    """

    metadata = _build_metadata(config=config)
    image_info = _build_image_info(info=info)

    if 'image_shape' in info:
        image_width = info['image_shape'][1]
    else:
        image_width = _get_image_width(
            coadd_image_path=info['image_path'],
            coadd_image_ext=info['image_ext'],
        )

    object_data = _build_object_data(
        central_size=coadd_config['central_size'],
        buffer_size=coadd_config['buffer_size'],
        image_width=image_width,
        psf_box_size=coadd_config['psf_box_size'],
        wcs=info['%s_wcs' % coadd_config['wcs_type']],
        position_offset=info['position_offset'])

    eu.ostools.makedirs_fromfile(meds_path)

    with StagedOutFile(meds_path + '.fz', tmpdir=tmpdir) as sf:

        staged_meds_path = sf.path[:-3]

        with fitsio.FITS(staged_meds_path, 'rw', clobber=True) as fits:
            _coadd_and_write_images(
                fits=fits,
                object_data=object_data,
                info=info,
                single_epoch_config=single_epoch_config,
                wcs=info['%s_wcs' % coadd_config['wcs_type']],
                position_offset=info['position_offset'],
                coadding_weight=coadd_config['coadding_weight'],
                seed=seed,
                fpack_pars=fpack_pars,
                tmpdir=tmpdir,
            )

            fits.write(metadata, extname=METADATA_EXTNAME)
            fits.write(image_info, extname=IMAGE_INFO_EXTNAME)

        # fpack it
        try:
            os.remove(staged_meds_path + '.fz')
        except FileNotFoundError:
            pass
        cmd = 'fpack %s' % staged_meds_path
        logger.info('fpacking:')
        logger.info("    command: '%s'" % cmd)
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception:
            pass
        else:
            if remove_fits_file:
                os.remove(staged_meds_path)

    # validate the fpacked file
    logger.info('validating:')
    validate_meds(meds_path + '.fz')


def _coadd_and_write_images(
        *, fits, fpack_pars, object_data, info, single_epoch_config,
        wcs, position_offset, coadding_weight, seed,
        tmpdir=None):

    logger.info('reserving mosaic images...')
    n_pixels = int(np.sum(object_data['box_size']**2))
    _reserve_images(fits, n_pixels, fpack_pars)

    rng = np.random.RandomState(seed=seed)
    n_psf_pixels = int(np.sum(object_data['psf_box_size']**2))

    # set the noise image seeds for each SE image via the RNG once
    for i in range(len(info['src_info'])):
        info['src_info'][i]['noise_seed'] = rng.randint(low=1, high=2**30)

    # some constraints
    # - we don't want to keep all of the data for each object in memory
    # - we also probably want to write the PSFs in another loop even though we
    #   we get them now (allows us to set the object data properly)
    # - we are going to get back a ton of SE bit masks which we may
    #   want to keep
    #
    # thus we do the following
    # - write the big slice images to disk right away
    # - keep the PSF images (smaller) in memory
    # - keep the single epoch bit masks in memory in a compressed format
    #
    # then at the end we write any daya we have in memory to disk
    psf_data = np.zeros(n_psf_pixels, dtype=CUTOUT_DTYPES[PSF_CUTOUT_EXTNAME])
    psf_data += CUTOUT_DEFAULT_VALUES[PSF_CUTOUT_EXTNAME]
    start_row = 0
    psf_start_row = 0
    epochs_info = []

    for i in prange(len(object_data)):
        logger.info('processing object %d', i)

        # we center the PSF at the nearest pixel center near the patch center
        col, row = wcs.sky2image(object_data['ra'][i], object_data['dec'][i])
        # this col, row includes the position offset
        # we don't need to remove it when putting them back into the WCS
        # but we will remove it later since we work in zero-indexed coords
        col = int(col + 0.5)
        row = int(row + 0.5)
        # ra, dec of the pixel center
        ra_psf, dec_psf = wcs.image2sky(col, row)

        # now we find the lower left location of the PSF image
        half = (object_data['psf_box_size'][i] - 1) / 2
        assert int(half) == half, "PSF images must have odd dimensions!"
        # here we remove the position offset
        col -= position_offset
        row -= position_offset
        psf_orig_start_col = col - half
        psf_orig_start_row = row - half

        bsres = _build_slice_inputs(
            ra=object_data['ra'][i],
            dec=object_data['dec'][i],
            ra_psf=ra_psf,
            dec_psf=dec_psf,
            box_size=object_data['box_size'][i],
            frac_buffer=single_epoch_config['frac_buffer'],
            coadd_info=info,
            start_row=object_data['orig_start_row'][i, 0],
            start_col=object_data['orig_start_col'][i, 0],
            se_src_info=info['src_info'],
            reject_outliers=single_epoch_config['reject_outliers'],
            symmetrize_masking=single_epoch_config['symmetrize_masking'],
            coadding_weight=coadding_weight,
            noise_interp_flags=sum(single_epoch_config['noise_interp_flags']),
            spline_interp_flags=sum(
                single_epoch_config['spline_interp_flags']),
            bad_image_flags=sum(single_epoch_config['bad_image_flags']),
            max_masked_fraction=single_epoch_config['max_masked_fraction'],
            max_unmasked_trail_fraction=single_epoch_config[
                'max_unmasked_trail_fraction'],
            mask_tape_bumps=single_epoch_config['mask_tape_bumps'],
            edge_buffer=single_epoch_config['edge_buffer'],
            wcs_type=single_epoch_config['wcs_type'],
            psf_type=single_epoch_config['psf_type'],
            rng=rng,
            tmpdir=tmpdir,
        )

        se_image_slices, weights, slices_not_used, flags_not_used = bsres

        logger.info('weights: %s' % weights)
        logger.info('using nepoch: %d' % len(weights))
        epochs_info.append(_make_epochs_info(
            object_data=object_data[i],
            weights=weights,
            slices=se_image_slices,
            slices_not_used=slices_not_used,
            flags_not_used=flags_not_used
        ))

        # did we get anything?
        if np.array(weights).size > 0:
            object_data['ncutout'][i] = 1
            object_data['nepoch'][i] = weights.size
            object_data['nepoch_eff'][i] = weights.sum()/weights.max()

            image, bmask, ormask, noise, psf, weight = _coadd_slice_inputs(
                wcs=wcs,
                wcs_position_offset=position_offset,
                start_row=object_data['orig_start_row'][i, 0],
                start_col=object_data['orig_start_col'][i, 0],
                box_size=object_data['box_size'][i],
                psf_start_row=psf_orig_start_row,
                psf_start_col=psf_orig_start_col,
                psf_box_size=object_data['psf_box_size'][i],
                se_image_slices=se_image_slices,
                weights=weights)

            if np.all(image == 0):
                logger.warning("coadded image is all zero!")

            # write the image, bmask, ormask, noise and weight map
            _write_single_image(
                fits=fits, data=image,
                ext=IMAGE_CUTOUT_EXTNAME, start_row=start_row)

            _write_single_image(
                fits=fits, data=bmask,
                ext=BMASK_CUTOUT_EXTNAME, start_row=start_row)

            _write_single_image(
                fits=fits, data=ormask,
                ext=ORMASK_CUTOUT_EXTNAME, start_row=start_row)

            _write_single_image(
                fits=fits, data=noise,
                ext=NOISE_CUTOUT_EXTNAME, start_row=start_row)

            _write_single_image(
                fits=fits, data=weight,
                ext=WEIGHT_CUTOUT_EXTNAME, start_row=start_row)

            # we need to keep the PSFs for writing later
            _psf_size = object_data['psf_box_size'][i]**2
            psf_data[psf_start_row:psf_start_row + _psf_size] = psf.ravel()

            object_data['psf_sigma'][i, 0] = measure_fwhm(psf)

            # now we need to set the start row so we know where the data is
            object_data['start_row'][i, 0] = start_row
            object_data['psf_start_row'][i, 0] = psf_start_row

            # finally we increment so that we have the next pixel for the
            # next sliee
            start_row += object_data['box_size'][i]**2
            psf_start_row += _psf_size

    fits.write(object_data, extname=OBJECT_DATA_EXTNAME)

    # we have accumulated the PSFs above and now we write them out
    fits.create_image_hdu(
        img=None,
        dtype=CUTOUT_DTYPES[PSF_CUTOUT_EXTNAME],
        dims=psf_data.shape,
        extname=PSF_CUTOUT_EXTNAME,
        header=fpack_pars)
    # also need to write the header...IDK why...
    fits[PSF_CUTOUT_EXTNAME].write_keys(fpack_pars, clean=False)
    fits[PSF_CUTOUT_EXTNAME].write(psf_data, start=0)

    epochs_info = eu.numpy_util.combine_arrlist(epochs_info)
    fits.write(epochs_info, extname=EPOCHS_INFO_EXTNAME)


def _reserve_images(fits, n_pixels, fpack_pars):
    """Does everything but the PSF image."""
    dims = [n_pixels]
    for ext in [
            IMAGE_CUTOUT_EXTNAME,
            WEIGHT_CUTOUT_EXTNAME,
            SEG_CUTOUT_EXTNAME,
            BMASK_CUTOUT_EXTNAME,
            ORMASK_CUTOUT_EXTNAME,
            NOISE_CUTOUT_EXTNAME]:
        fits.create_image_hdu(
            img=None,
            dtype=CUTOUT_DTYPES[ext],
            dims=dims,
            extname=ext,
            header=fpack_pars)

        # also need to write the header...IDK why...
        fits[ext].write_keys(fpack_pars, clean=False)


def _write_single_image(*, fits, data, ext, start_row):
    subim = np.zeros(data.shape, dtype=CUTOUT_DTYPES[ext])
    subim += CUTOUT_DEFAULT_VALUES[ext]
    subim[:, :] = data
    # TODO: do I need to add .ravel() here?
    fits[ext].write(subim, start=start_row)


@functools.lru_cache(maxsize=2048)
def _get_image_width(*, coadd_image_path, coadd_image_ext):
    coadd_image_path = expandvars(coadd_image_path)
    h = fitsio.read_header(coadd_image_path, ext=coadd_image_ext)
    if 'znaxis1' in h:
        return h['znaxis1']
    else:
        return h['naxis1']


def _make_epochs_info(
        *, object_data, weights, slices, slices_not_used, flags_not_used):
    """Record info for each epoch we considered for a given slice.

    Parameters
    ----------
    object_data : np.ndarray
        The object_data entry for the slice.
    weights : np.ndarray
        The weights for used images, can be zero length.
    slices : list of SEImageSlices
        The image slices used for the coadd.
    slices_not_used : list of SEImageSlices
        The image slices not used for the coadd.
    flags_not_used : list of SEImageSlices
        The flag values for the slices not used.

    Returns
    -------
    epoch_info : structured np.ndarray
        A structured array with the epoch info.
    """
    dt = [
        ('id', 'i8'),
        ('file_id', 'i8'),  # index into image info
        ('flags', 'i4'),
        ('row_start', 'i8'),
        ('col_start', 'i8'),
        ('box_size', 'i8'),
        ('psf_row_start', 'i8'),
        ('psf_col_start', 'i8'),
        ('psf_box_size', 'i8'),
        ('weight', 'f8'),
    ]

    data = np.zeros(len(slices) + len(slices_not_used), dtype=dt)
    data['id'] = object_data['id']

    loc = 0

    for weight, se_slice in zip(weights, slices):
        data['flags'][loc] = 0  # we used it, so flags are zero
        data['file_id'][loc] = se_slice.source_info['file_id']

        data['row_start'][loc] = se_slice.y_start
        data['col_start'][loc] = se_slice.x_start
        data['box_size'][loc] = se_slice.box_size

        data['psf_row_start'][loc] = se_slice.psf_y_start
        data['psf_col_start'][loc] = se_slice.psf_x_start
        data['psf_box_size'][loc] = se_slice.psf_box_size
        data['weight'][loc] = weight

        loc += 1

    for flags, se_slice in zip(flags_not_used, slices_not_used):
        data['flags'][loc] = flags
        data['file_id'][loc] = se_slice.source_info['file_id']

        data['row_start'][loc] = se_slice.y_start
        data['col_start'][loc] = se_slice.x_start
        data['box_size'][loc] = se_slice.box_size

        data['psf_row_start'][loc] = se_slice.psf_y_start
        data['psf_col_start'][loc] = se_slice.psf_x_start
        data['psf_box_size'][loc] = se_slice.psf_box_size
        data['weight'][loc] = 0.0  # we did not use this slice, so zero

        loc += 1

    return data


def _build_object_data(
        *,
        central_size,
        buffer_size,
        image_width,
        psf_box_size,
        wcs,
        position_offset):
    """Build the internal MEDS data structure.

    NOTE: Information about the PSF is filled when the PSF is drawn later.
    """
    rows, cols, start_rows, start_cols = build_slice_locations(
        central_size=central_size,
        buffer_size=buffer_size,
        image_width=image_width)
    box_size = central_size + 2 * buffer_size
    cen_offset = (central_size - 1) / 2.0  # zero-indexed at pixel centers
    psf_cen = (psf_box_size - 1) / 2

    # now make the object data and extra fields for the PSF
    # we use an namx of 2 because if we set 1, then the fields come out with
    # the wrong shape when we read them from fitsio
    nmax = 2
    psf_dtype = [
        ('psf_box_size', 'i4'),
        ('psf_cutout_row', 'f8', nmax),
        ('psf_cutout_col', 'f8', nmax),
        ('psf_sigma', 'f4', nmax),
        ('psf_start_row', 'i8', nmax),
    ]
    # extra metadata not stored in standard meds files
    meta_extra = [
        ('nepoch', 'i4'),
        ('nepoch_eff', 'f8'),
    ]
    output_info = get_meds_output_struct(
        len(rows),
        nmax,
        extra_fields=meta_extra + psf_dtype,
    )

    # and fill!
    output_info['id'] = np.arange(len(rows))
    output_info['box_size'] = box_size
    output_info['file_id'] = 0
    output_info['psf_box_size'] = psf_box_size
    output_info['psf_cutout_row'] = psf_cen
    output_info['psf_cutout_col'] = psf_cen

    for index, (row, col, start_row, start_col) in enumerate(zip(
            rows, cols, start_rows, start_cols)):
        ra, dec = wcs.image2sky(
            x=col + position_offset,
            y=row + position_offset)
        jacob = wcs.get_jacobian(
            x=col + position_offset,
            y=row + position_offset)

        output_info['ra'][index] = ra
        output_info['dec'][index] = dec
        output_info['orig_row'][index, 0] = row
        output_info['orig_col'][index, 0] = col
        output_info['orig_start_row'][index, 0] = start_row
        output_info['orig_start_col'][index, 0] = start_col
        output_info['cutout_row'][index, 0] = cen_offset + buffer_size
        output_info['cutout_col'][index, 0] = cen_offset + buffer_size
        output_info['dudcol'][index, 0] = jacob[0]
        output_info['dudrow'][index, 0] = jacob[1]
        output_info['dvdcol'][index, 0] = jacob[2]
        output_info['dvdrow'][index, 0] = jacob[3]

    return output_info


def _build_image_info(*, info):
    n_images = 1 + len(info['src_info'])

    # we need to get the maximum WCS length here
    max_wcs_len = max(
        [len(json.dumps(eval(str(info['image_wcs']))))] + [
            len(json.dumps(eval(str(se['image_wcs']))))
            for se in info['src_info']])

    # we need to get the maximum string length here too
    max_str_len = max([
        max([
            len(se['image_path']),
            len(se['image_ext']),
            len(se['bkg_path']),
            len(se['bkg_ext']),
            len(se['weight_path']),
            len(se['weight_ext']),
            len(se['bmask_path']),
            len(se['bmask_ext'])])
        for se in info['src_info']])

    ii = get_image_info_struct(
        n_images,
        max_str_len,
        wcs_len=max_wcs_len,
        ext_len=3)

    # first we do the coadd since it is special
    ii['image_id'][0] = info['file_id']
    ii['image_flags'][0] = info['image_flags']
    ii['magzp'][0] = info['magzp']
    ii['scale'][0] = info['scale']
    ii['position_offset'][0] = info['position_offset']
    ii['wcs'][0] = json.dumps(eval(str(info['image_wcs'])))

    # now do the epochs
    for i, se_info in enumerate(info['src_info']):
        loc = i + 1
        for key in [
                'image_path', 'image_ext', 'weight_path', 'weight_ext',
                'bmask_path', 'bmask_ext', 'bkg_path', 'bkg_ext']:
            ii[key][loc] = se_info[key]
        ii['image_id'][loc] = se_info['file_id']
        ii['image_flags'][loc] = se_info['image_flags']
        ii['magzp'][loc] = se_info['magzp']
        ii['scale'][loc] = se_info['scale']
        ii['position_offset'][loc] = se_info['position_offset']
        ii['wcs'][loc] = json.dumps(eval(str(se_info['image_wcs'])))

    return ii


def _build_metadata(*, config):
    numpy_version = np.__version__
    esutil_version = eu.__version__
    fitsio_version = fitsio.__version__
    meds_version = meds.__version__
    piff_version = piff.__version__
    pixmappy_version = pixmappy.__version__
    desmeds_version = desmeds.__version__
    dt = [
        ('magzp_ref', 'f8'),
        ('config', 'S%d' % len(config)),
        ('pizza_cutter_version', 'S%d' % len(__version__)),
        ('numpy_version', 'S%d' % len(numpy_version)),
        ('esutil_version', 'S%d' % len(esutil_version)),
        ('fitsio_version', 'S%d' % len(fitsio_version)),
        ('piff_version', 'S%d' % len(piff_version)),
        ('pixmappy_version', 'S%d' % len(pixmappy_version)),
        ('desmeds_version', 'S%d' % len(desmeds_version)),
        ('meds_version', 'S%d' % len(meds_version)),
        ('meds_fmt_version', 'S%d' % len(MEDS_FMT_VERSION)),
        ('meds_dir', 'S%d' % len(os.environ['MEDS_DIR'])),
        ('piff_data_dir', 'S%d' % len(os.environ.get('PIFF_DATA_DIR', ' '))),
        ('desdata', 'S%d' % len(os.environ.get('DESDATA', ' ')))]
    metadata = np.zeros(1, dt)
    metadata['magzp_ref'] = MAGZP_REF
    metadata['config'] = config
    metadata['numpy_version'] = numpy_version
    metadata['esutil_version'] = esutil_version
    metadata['fitsio_version'] = fitsio_version
    metadata['piff_version'] = piff_version
    metadata['pixmappy_version'] = pixmappy_version
    metadata['desmeds_version'] = desmeds_version
    metadata['meds_version'] = meds_version
    metadata['meds_fmt_version'] = MEDS_FMT_VERSION
    metadata['pizza_cutter_version'] = __version__
    metadata['meds_dir'] = os.environ['MEDS_DIR']
    metadata['piff_data_dir'] = os.environ.get('PIFF_DATA_DIR', '')
    metadata['desdata'] = os.environ.get('DESDATA', '')
    return metadata
