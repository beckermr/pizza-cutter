import os
from os.path import expandvars
import subprocess
import functools
import json
import hashlib
import logging
import multiprocessing as mp
import copy
from functools import lru_cache

import numpy as np
import fitsio
import meds
import esutil as eu
import piff
import pixmappy
import desmeds
import ngmix
import scipy
from hilbertcurve.hilbertcurve import HilbertCurve

from metadetect.masking import make_foreground_bmask

from meds.maker import MEDS_FMT_VERSION
from meds.util import (
    get_image_info_struct, get_meds_output_struct, validate_meds)

from .. import __version__
from ._se_image import _load_image_wcs
from ._constants import (
    METADATA_EXTNAME,
    GAIA_STARS_EXTNAME,
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
    MFRAC_CUTOUT_EXTNAME,
    CUTOUT_DTYPES,
    CUTOUT_DEFAULT_VALUES,
    TILE_INFO_EXTNAME,
)
from ..slice_utils.locate import build_slice_locations
from ..slice_utils.measure import measure_fwhm
from ..files import StagedOutFile
from ._coadd_slices import (
    _build_slice_inputs, _coadd_slice_inputs)

from esutil.pbar import PBar

logger = logging.getLogger(__name__)


RESULT_QUEUE = None


def _hash_to_int(s):
    # from https://stackoverflow.com/questions/
    #  16008670/how-to-hash-a-string-into-8-digits
    return min(
            max(1, abs(int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)) % 10**4),
            9999,
        )


def _init_result_queue(q):
    global RESULT_QUEUE
    RESULT_QUEUE = q


def make_des_pizza_slices(
    *,
    config,
    meds_path,
    info,
    json_info,
    seed,
    slice_range=None,
    remove_fits_file=True,
    tmpdir=None,
    fpack_pars=None,
    coadd_config,
    single_epoch_config,
    n_extra_noise_images,
    n_jobs=1,
    n_chunks=None,
):
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
    json_info : str
        The original info file as json. This will be written to the metadata.
    seed : int
        The random seed used to make the noise field.
    slice_range: iterable or str
        Range of slices to process.  Either an iterable, e.g. range(10, 20) or
        a string of the form 'start:end' representing a python slice.  Default
        None which means process all slices.
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
    n_extra_noise_images : int
        The number of extra noise images to make. These are written as cutout
        types 'noise1', 'noise2', etc. in the final MEDS file.
    n_jobs : int, optional
        The number of multiprocessing jobs to use. Only works well for large
        numbers of slices.
    n_chunks : int, optional
        The number of chunks to use with n_jobs. Defaults to n_jobs if not set.
    """

    metadata, json_info_image = _build_metadata(config=config, json_info=json_info)
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

    wcs = info['%s_wcs' % coadd_config['wcs_type']]
    position_offset = info['position_offset']

    gaia_stars_file = info.get('gaia_stars_file', None)

    with StagedOutFile(meds_path + '.fz', tmpdir=tmpdir) as sf:

        staged_meds_path = sf.path[:-3]

        with fitsio.FITS(staged_meds_path, 'rw', clobber=True) as fits:
            _coadd_and_write_images(
                fits=fits,
                object_data=object_data,
                info=info,
                single_epoch_config=single_epoch_config,
                wcs=wcs,
                position_offset=position_offset,
                coadding_weight=coadd_config['coadding_weight'],
                seed=seed,
                slice_range=slice_range,
                fpack_pars=fpack_pars,
                tmpdir=tmpdir,
                n_jobs=n_jobs,
                n_chunks=n_chunks,
                n_extra_noise_images=n_extra_noise_images,
            )

            print("writing metadata", flush=True)
            fits.write(metadata, extname=METADATA_EXTNAME)
            fits.write(json_info_image, extname=TILE_INFO_EXTNAME)
            fits.write(image_info, extname=IMAGE_INFO_EXTNAME)

            if gaia_stars_file is not None:
                gaia_stars = _read_gaia_stars(
                    fname=gaia_stars_file,
                    wcs=wcs,
                    wcs_position_offset=position_offset,
                )
                fits.write(gaia_stars, extname=GAIA_STARS_EXTNAME)

        # fpack it
        try:
            os.remove(staged_meds_path + '.fz')
        except FileNotFoundError:
            pass
        fpack_seed = _hash_to_int(
            info["tilename"] + info["band"]
        )
        cmd = 'fpack -qz%d 16 %s' % (fpack_seed, staged_meds_path)
        print("fpacking:", flush=True)
        print("    command: '%s'" % cmd, flush=True)
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception:
            pass
        else:
            if remove_fits_file:
                os.remove(staged_meds_path)

    # validate the fpacked file
    print('validating:', flush=True)
    validate_meds(meds_path + '.fz')


def _add_gaia_radius(*, gaia_stars, max_g_mag, poly_coeffs):
    w, = np.where(gaia_stars['phot_g_mean_mag'] <= max_g_mag)
    gaia_stars = gaia_stars[w]

    add_dt = [('radius_pixels', 'f4')]
    gaia_stars = eu.numpy_util.add_fields(gaia_stars, add_dt)

    ply = np.poly1d(poly_coeffs)
    log10_radius_pixels = ply(gaia_stars['phot_g_mean_mag'])
    gaia_stars['radius_pixels'] = 10.0**log10_radius_pixels
    return gaia_stars


def _build_gaia_star_mask(
    *, gaia_stars, max_g_mag, poly_coeffs, start_col, start_row, box_size, symmetrize
):
    gaia_stars = _add_gaia_radius(
        gaia_stars=gaia_stars,
        max_g_mag=max_g_mag,
        poly_coeffs=poly_coeffs,
    )

    return make_foreground_bmask(
        xm=gaia_stars['x'].astype('f8') - start_col,
        ym=gaia_stars['y'].astype('f8') - start_row,
        rm=gaia_stars['radius_pixels'].astype('f8'),
        dims=(box_size, box_size),
        symmetrize=symmetrize,
        mask_bit_val=2**0,
    ).astype(bool)


def _build_coadd_psf_layout(*, wcs, ra, dec, box_size, position_offset):
    # we center the PSF at the nearest pixel center near the patch center
    col, row = wcs.sky2image(ra, dec)
    # this col, row includes the position offset
    # we don't need to remove it when putting them back into the WCS
    # but we will remove it later since we work in zero-indexed coords
    col = int(np.floor(col + 0.5))
    row = int(np.floor(row + 0.5))
    # ra, dec of the pixel center
    ra_psf, dec_psf = wcs.image2sky(col, row)

    # now we find the lower left location of the PSF image
    half = (box_size - 1) / 2
    assert int(half) == half, "PSF images must have odd dimensions!"
    # here we remove the position offset
    col -= position_offset
    row -= position_offset
    psf_orig_start_col = col - half
    psf_orig_start_row = row - half

    return ra_psf, dec_psf, psf_orig_start_row, psf_orig_start_col


def _coadd_single_slice(
    *, i, object_data, info, single_epoch_config, wcs, position_offset,
    coadding_weight, slice_seed, tmpdir, n_extra_noise_images,
):
    logger.info('processing slice %d', i)

    results = {'i': i}
    rng = np.random.RandomState(seed=slice_seed)

    ra_psf, dec_psf, psf_orig_start_row, psf_orig_start_col = _build_coadd_psf_layout(
        wcs=wcs,
        ra=object_data["ra"][i],
        dec=object_data["dec"][i],
        box_size=object_data["psf_box_size"][i],
        position_offset=position_offset,
    )

    gaia_stars_file = info.get('gaia_stars_file', None)
    if gaia_stars_file is not None and "gaia_star_masks" in single_epoch_config:
        logger.info("building GAIA star mask for slice %d", i)
        gaia_stars = _read_gaia_stars(
            fname=gaia_stars_file,
            wcs=wcs,
            wcs_position_offset=position_offset,
        )
        gaia_star_mask = _build_gaia_star_mask(
            gaia_stars=gaia_stars,
            max_g_mag=single_epoch_config["gaia_star_masks"]["max_g_mag"],
            poly_coeffs=single_epoch_config["gaia_star_masks"]["poly_coeffs"],
            start_col=object_data['orig_start_col'][i, 0],
            start_row=object_data['orig_start_row'][i, 0],
            box_size=object_data['box_size'][i],
            symmetrize=single_epoch_config["gaia_star_masks"]["symmetrize"]
        )
    else:
        gaia_star_mask = None

    bsres = _build_slice_inputs(
        ra=object_data['ra'][i],
        dec=object_data['dec'][i],
        ra_psf=ra_psf,
        dec_psf=dec_psf,
        box_size=object_data['box_size'][i],
        frac_buffer=single_epoch_config['frac_buffer'],
        start_row=object_data['orig_start_row'][i, 0],
        start_col=object_data['orig_start_col'][i, 0],
        wcs=wcs,
        wcs_position_offset=position_offset,
        wcs_interp_delta=single_epoch_config["se_wcs_interp_delta"],
        gaia_star_mask=gaia_star_mask,
        se_src_info=info['src_info'],
        reject_outliers=single_epoch_config['reject_outliers'],
        symmetrize_masking=single_epoch_config['symmetrize_masking'],
        coadding_weight=coadding_weight,
        copy_masked_edges=single_epoch_config['copy_masked_edges'],
        noise_interp_flags=sum(single_epoch_config['noise_interp_flags']),
        spline_interp_flags=sum(
            single_epoch_config['spline_interp_flags']),
        bad_image_flags=sum(single_epoch_config['bad_image_flags']),
        max_masked_fraction=single_epoch_config['max_masked_fraction'],
        mask_tape_bumps=single_epoch_config['mask_tape_bumps'],
        mask_piff_failure_config=single_epoch_config["mask_piff_failure"],
        edge_buffer=single_epoch_config['edge_buffer'],
        wcs_type=single_epoch_config['wcs_type'],
        wcs_color=single_epoch_config['wcs_color'],
        psf_type=single_epoch_config['psf_type'],
        psf_kwargs=single_epoch_config['psf_kwargs'][info["band"]],
        rng=rng,
        tmpdir=tmpdir,
        n_extra_noise_images=n_extra_noise_images,
    )

    se_image_slices, weights, slices_not_used, flags_not_used = bsres

    logger.info('weights: %s' % weights)
    logger.info('using nepoch: %d' % len(weights))
    results["epochs_info"] = _make_epochs_info(
        object_data=object_data[i],
        weights=weights,
        slices=se_image_slices,
        slices_not_used=slices_not_used,
        flags_not_used=flags_not_used
    )
    results["weights"] = weights

    # did we get anything?
    if np.array(weights).size > 0:
        (
            image, bmask, ormask, noises, psf, weight, mfrac, _
        ) = _coadd_slice_inputs(
            wcs=wcs,
            wcs_position_offset=position_offset,
            wcs_image_shape=info["image_shape"],
            start_row=object_data['orig_start_row'][i, 0],
            start_col=object_data['orig_start_col'][i, 0],
            box_size=object_data['box_size'][i],
            psf_start_row=psf_orig_start_row,
            psf_start_col=psf_orig_start_col,
            psf_box_size=object_data['psf_box_size'][i],
            se_image_slices=se_image_slices,
            weights=weights,
            se_wcs_interp_delta=single_epoch_config["se_wcs_interp_delta"],
            coadd_wcs_interp_delta=single_epoch_config["coadd_wcs_interp_delta"],
            n_extra_noise_images=n_extra_noise_images,
        )

        if np.all(image == 0):
            logger.warning("coadded image is all zero!")

        results["image"] = image
        results["bmask"] = bmask
        results["ormask"] = ormask
        results["noises"] = noises
        results["psf"] = psf
        results["weight"] = weight
        results["mfrac"] = mfrac
        results["psf_sigma"] = measure_fwhm(psf)

    return results


def _process_slice_chunk(
    *, slice_inds, object_data, info, single_epoch_config, wcs, position_offset,
    coadding_weight, slice_seeds, tmpdir, n_extra_noise_images,
):
    try:
        for i, slice_seed in zip(slice_inds, slice_seeds):
            results = _coadd_single_slice(
                i=i,
                object_data=object_data,
                info=info,
                single_epoch_config=single_epoch_config,
                wcs=wcs,
                position_offset=position_offset,
                coadding_weight=coadding_weight,
                slice_seed=slice_seed,
                tmpdir=tmpdir,
                n_extra_noise_images=n_extra_noise_images,
            )
            RESULT_QUEUE.put(results, block=True)
    except Exception as e:
        import traceback
        eret = RuntimeError(
            f"Encountered error for slice {i} w/ seed {slice_seed}:\n"
            f"error: {repr(e)}\n"
            f"traceback: {traceback.format_exc()}"
        )
        RESULT_QUEUE.put(eret, block=True)


def _process_results(
    *, results, start_row, psf_start_row, object_data, epochs_info, fits,
):
    logger.info("writing data for slice %s", results['i'])
    slice_id = results['i']
    epochs_info.append(results["epochs_info"])

    # did we get anything?
    if np.array(results["weights"]).size > 0:
        object_data['ncutout'][slice_id] = 1
        object_data['nepoch'][slice_id] = results["weights"].size
        object_data['nepoch_eff'][slice_id] = (
            results["weights"].sum() / results["weights"].max()
        )

        # write the image, bmask, ormask, noise and weight map
        _write_single_image(
            fits=fits, data=results["image"],
            ext=IMAGE_CUTOUT_EXTNAME, start_row=start_row)

        _write_single_image(
            fits=fits, data=results["bmask"],
            ext=BMASK_CUTOUT_EXTNAME, start_row=start_row)

        _write_single_image(
            fits=fits, data=results["ormask"],
            ext=ORMASK_CUTOUT_EXTNAME, start_row=start_row)

        _write_single_image(
            fits=fits, data=results["noises"][0],
            ext=NOISE_CUTOUT_EXTNAME, start_row=start_row)
        if len(results["noises"]) > 1:
            for nse_i in range(1, len(results["noises"])):
                _write_single_image(
                    fits=fits, data=results["noises"][nse_i],
                    ext="noise%d_cutouts" % nse_i, start_row=start_row,
                    ext_info=NOISE_CUTOUT_EXTNAME,
                )

        _write_single_image(
            fits=fits, data=results["weight"],
            ext=WEIGHT_CUTOUT_EXTNAME, start_row=start_row)

        _write_single_image(
            fits=fits, data=results["mfrac"],
            ext=MFRAC_CUTOUT_EXTNAME, start_row=start_row)

        _write_single_image(
            fits=fits, data=results["psf"],
            ext=PSF_CUTOUT_EXTNAME, start_row=psf_start_row)

        object_data['psf_sigma'][slice_id, 0] = results["psf_sigma"]

        # now we need to set the start row so we know where the data is
        object_data['start_row'][slice_id, 0] = start_row
        object_data['psf_start_row'][slice_id, 0] = psf_start_row
    else:
        object_data['nepoch'][slice_id] = 0
        object_data['nepoch_eff'][slice_id] = 0


def _coadd_and_write_images(
    *, fits, fpack_pars, object_data, info, single_epoch_config,
    wcs, position_offset, coadding_weight, seed,
    slice_range=None,
    tmpdir=None, n_jobs=1, n_extra_noise_images, n_chunks=None,
):

    if n_chunks is None:
        n_chunks = n_jobs

    # we use a space-filling curve to order the slices
    # avoids cache misses for the internal LRU caches
    slices_to_do = _extract_slice_range(
        slice_range=slice_range,
        num=len(object_data),
    )
    slices_to_do = np.array(list(slices_to_do), dtype=int)
    ph_order = 8
    hilbert_curve = HilbertCurve(ph_order, 2)
    distances = []
    dx = info['image_shape'][1]/ph_order
    dy = info['image_shape'][0]/ph_order
    print("ordering slices for better cache use", flush=True)
    for i in slices_to_do:
        col, row = wcs.sky2image(object_data['ra'][i], object_data['dec'][i])
        xind = min(max(int(col / dx), 0), ph_order-1)
        yind = min(max(int(row / dy), 0), ph_order-1)
        distances.append(hilbert_curve.distance_from_point([xind, yind]))
    sind = np.argsort(distances)
    slices_to_do = [s for s in slices_to_do[sind]]
    n_slices_to_do = len(slices_to_do)

    logger.info('reserving mosaic images...')
    npix = object_data['box_size'][0]**2
    npix_psf = object_data['psf_box_size'][0]**2
    n_pixels = int(np.sum(object_data['box_size']**2))
    n_pixels_psf = int(np.sum(object_data['psf_box_size']**2))
    _reserve_images(
        fits, n_pixels, n_pixels_psf, fpack_pars, n_extra_noise_images,
        npix, npix_psf
    )

    rng = np.random.RandomState(seed=seed)
    slice_seeds = rng.randint(1, 2**32-1, size=len(object_data))

    # set the noise image seeds for each SE image via the RNG once
    # also set a seed for masking
    for i in range(len(info['src_info'])):
        info['src_info'][i]['noise_seeds'] = rng.randint(
            low=1, high=2**30, size=n_extra_noise_images+1)
        info['src_info'][i]["mask_piff_failure_seed"] = rng.randint(low=1, high=2**30)

    print(
        'processing %d slices: %s' % (len(slices_to_do), slice_range or "all slices"),
        flush=True,
    )

    epochs_info = []

    if n_jobs > 1:
        nsub = n_slices_to_do // n_chunks
        if nsub * n_chunks < n_slices_to_do:
            nsub += 1

        result_queue = mp.Queue(maxsize=10*n_jobs)
        with mp.Pool(
            processes=n_jobs,
            initializer=_init_result_queue,
            initargs=(result_queue,),
        ) as exec:
            jobs = [[] for _ in range(n_chunks)]
            worker_seeds = [[] for _ in range(n_chunks)]
            futs = []
            for w in range(n_chunks):
                for s in range(nsub):
                    loc = w * nsub + s
                    if loc < len(slices_to_do):
                        jobs[w].append(slices_to_do[loc])
                        worker_seeds[w].append(slice_seeds[jobs[w][-1]])

            print("job chunk legths:", [len(j) for j in jobs], flush=True)

            for w in range(n_chunks):
                futs.append(exec.apply_async(
                    _process_slice_chunk,
                    kwds=dict(
                        slice_inds=jobs[w],
                        object_data=object_data,
                        info=info,
                        single_epoch_config=single_epoch_config,
                        wcs=wcs,
                        position_offset=position_offset,
                        coadding_weight=coadding_weight,
                        slice_seeds=worker_seeds[w],
                        tmpdir=tmpdir,
                        n_extra_noise_images=n_extra_noise_images,
                    ),
                ))

            for ib in PBar(range(n_slices_to_do), total=n_slices_to_do):
                logger.debug("waiting for result %d", ib)
                results = result_queue.get()
                if isinstance(results, Exception):
                    raise results
                _process_results(
                    results=results,
                    start_row=results['i'] * npix,
                    psf_start_row=results['i'] * npix_psf,
                    object_data=object_data,
                    epochs_info=epochs_info,
                    fits=fits,
                )

            [fut.get() for fut in futs]
    else:
        for i in PBar(slices_to_do, total=n_slices_to_do):
            results = _coadd_single_slice(
                i=i,
                object_data=object_data,
                info=info,
                single_epoch_config=single_epoch_config,
                wcs=wcs,
                position_offset=position_offset,
                coadding_weight=coadding_weight,
                slice_seed=slice_seeds[i],
                tmpdir=tmpdir,
                n_extra_noise_images=n_extra_noise_images,
            )
            _process_results(
                results=results,
                start_row=results['i'] * npix,
                psf_start_row=results['i'] * npix_psf,
                object_data=object_data,
                epochs_info=epochs_info,
                fits=fits,
            )

    fits.write(object_data, extname=OBJECT_DATA_EXTNAME)

    assert len(epochs_info) == n_slices_to_do
    epochs_info = eu.numpy_util.combine_arrlist(epochs_info)
    fits.write(epochs_info, extname=EPOCHS_INFO_EXTNAME)


@lru_cache(maxsize=1)
def _read_gaia_stars(
    fname,
    wcs,
    wcs_position_offset,
):
    """
    load the gaia stars

    Parameters
    -----------
    fname: str
        path to the gaia star catalog
    wcs: WCS object
        The WCS object for the image. Used to calculate x, y
    wcs_position_offset : int
        The position offset to get from zero-indexed, pixel-centered
        coordinates to the coordinates expected by the coadd WCS object.
    """

    full_path = os.path.expandvars(fname)
    logger.info('reading: %s' % full_path)
    data = fitsio.read(full_path, lower=True)

    add_dt = [('x', 'f8'), ('y', 'f8')]
    data = eu.numpy_util.add_fields(data, add_dt)

    data['x'], data['y'] = wcs.sky2image(data['ra'], data['dec'])

    data['x'] -= wcs_position_offset
    data['y'] -= wcs_position_offset

    return data


def _reserve_images(
    fits, n_pixels, n_pixels_psf, fpack_pars, n_extra_noise_images,
    n_pixels_per, n_pixels_psf_per,
):
    names = [
        IMAGE_CUTOUT_EXTNAME,
        WEIGHT_CUTOUT_EXTNAME,
        SEG_CUTOUT_EXTNAME,
        BMASK_CUTOUT_EXTNAME,
        ORMASK_CUTOUT_EXTNAME,
        NOISE_CUTOUT_EXTNAME,
        MFRAC_CUTOUT_EXTNAME,
    ]
    extra_noise_names = [
        "noise%d_cutouts" % (i+1)
        for i in range(n_extra_noise_images)
    ]
    names += extra_noise_names
    dims = [n_pixels] * len(names) + [n_pixels_psf]
    dims_per = [n_pixels_per] * len(names) + [n_pixels_psf_per]
    names += [PSF_CUTOUT_EXTNAME]
    for ext, _dims, _dims_per in PBar(
        zip(names, dims, dims_per),
        total=len(names),
        desc='reserving image HDUs'
    ):
        if ext in extra_noise_names:
            # for these, use the info from the orig noise cutout
            ext_info = NOISE_CUTOUT_EXTNAME
        else:
            ext_info = ext

        fpp = copy.deepcopy(fpack_pars)

        if fpp is not None and "FZTILE" not in fpp:
            fpp["FZTILE"] = "(%d,1)" % _dims_per

        fits.create_image_hdu(
            img=None,
            dtype=CUTOUT_DTYPES[ext_info],
            dims=_dims,
            extname=ext,
            header=fpp)

        # also need to write the header...IDK why...
        if fpp is not None:
            fits[ext].write_keys(fpp, clean=False)


def _write_single_image(*, fits, data, ext, start_row, ext_info=None):
    if ext_info is None:
        ext_info = ext

    subim = np.zeros(data.shape, dtype=CUTOUT_DTYPES[ext_info])
    subim += CUTOUT_DEFAULT_VALUES[ext_info]
    subim[:, :] = data
    if ext in [MFRAC_CUTOUT_EXTNAME, WEIGHT_CUTOUT_EXTNAME]:
        if np.any(subim < 0):
            raise RuntimeError("negative values in ext %s: %f" % (ext, np.min(subim)))
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
        ('image_id', 'i8'),
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
        data['image_id'][loc] = se_slice.source_info['image_id']

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
        data['image_id'][loc] = se_slice.source_info['image_id']

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
    # this is not used here so set to -1
    output_info['file_id'] = -1
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


def _noerr_load_image_wcs(se):
    try:
        if 'image_wcs' in se:
            return se['image_wcs']
        else:
            return _load_image_wcs(se["image_path"], se["image_ext"])
    except Exception as e:
        if "affine_wcs_config" not in se:
            raise e
        else:
            return None


def _build_image_info(*, info):
    n_images = 1 + len(info['src_info'])

    # we need to get the maximum WCS length here
    max_wcs_len = max(
        [len(json.dumps(eval(str(info['image_wcs']))))]
        + [
            len(json.dumps(eval(str(
                _noerr_load_image_wcs(se)
            ))))
            for se in PBar(
                info['src_info'],
                total=len(info['src_info']),
                desc='making image info data',
            )
        ]
    )

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
    ii['image_id'][0] = info['image_id']
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
        ii['image_id'][loc] = se_info['image_id']
        ii['image_flags'][loc] = se_info['image_flags']
        ii['magzp'][loc] = se_info['magzp']
        ii['scale'][loc] = se_info['scale']
        ii['position_offset'][loc] = se_info['position_offset']
        ii['wcs'][loc] = json.dumps(eval(str(
            _noerr_load_image_wcs(se_info)
        )))

    assert np.array_equal(ii["image_id"], np.arange(len(ii), dtype=np.int32))

    return ii


def _build_metadata(*, config, json_info):
    numpy_version = np.__version__
    scipy_version = scipy.__version__
    esutil_version = eu.__version__
    fitsio_version = fitsio.__version__
    meds_version = meds.__version__
    piff_version = piff.__version__
    pixmappy_version = pixmappy.__version__
    desmeds_version = desmeds.__version__
    ngmix_version = ngmix.__version__
    dt = [
        ('magzp_ref', 'f8'),
        ('config', 'U%d' % len(config)),
        ('pizza_cutter_version', 'U%d' % len(__version__)),
        ('numpy_version', 'U%d' % len(numpy_version)),
        ('scipy_version', 'U%d' % len(scipy_version)),
        ('esutil_version', 'U%d' % len(esutil_version)),
        ('ngmix_version', 'U%d' % len(ngmix_version)),
        ('fitsio_version', 'U%d' % len(fitsio_version)),
        ('piff_version', 'U%d' % len(piff_version)),
        ('pixmappy_version', 'U%d' % len(pixmappy_version)),
        ('desmeds_version', 'U%d' % len(desmeds_version)),
        ('meds_version', 'U%d' % len(meds_version)),
        ('meds_fmt_version', 'U%d' % len(MEDS_FMT_VERSION)),
        ('meds_dir', 'U%d' % len(os.environ['MEDS_DIR'])),
        ('piff_data_dir', 'U%d' % len(os.environ.get('PIFF_DATA_DIR', ' '))),
        ('desdata', 'U%d' % len(os.environ.get('DESDATA', ' ')))]
    metadata = np.zeros(1, dt)
    metadata['magzp_ref'] = MAGZP_REF
    metadata['config'] = config
    metadata['numpy_version'] = numpy_version
    metadata['scipy_version'] = scipy_version
    metadata['esutil_version'] = esutil_version
    metadata['ngmix_version'] = ngmix_version
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
    return metadata, np.frombuffer(json_info.encode("ascii"), dtype='u1')


def _extract_slice_range(slice_range, num):
    if slice_range is None:
        ret = range(num)
    else:

        try:
            # try to split as a string
            ss = slice_range.split(':')
            assert len(ss) == 2
            start = int(ss[0])
            end = int(ss[1])
            ret = range(start, end)
        except AttributeError:
            try:
                # see if it is an iterable
                len(slice_range)
                ret = slice_range
            except TypeError:
                raise ValueError(
                    'bad slice specification: "%s"' % str(slice_range)
                )
        if ret.start < 0 or ret.stop > num:
            raise ValueError(
                'slice_range %s out of bounds [0, %d)' % (ret, num)
            )

    return ret
