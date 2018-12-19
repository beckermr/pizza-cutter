import os
import json
import time
import datetime
import logging

import joblib
import numpy as np
import esutil as eu
import fitsio

from metadetect.metadetect import do_metadetect
from metadetect.metadetect_and_cal import do_metadetect_and_cal

logger = logging.getLogger(__name__)


def _make_output_array(
        *,
        data, obj_id, mcal_step,
        orig_start_row, orig_start_col, position_offset, wcs,
        buffer_size, image_size):
    arr = eu.numpy_util.add_fields(
                data,
                [('id', 'i8'), ('mcal_step', 'S7'),
                 ('ra', 'f8'), ('dec', 'f8'),
                 ('mcal_ra', 'f8'), ('mcal_dec', 'f8'),
                 ('slice_flags', 'i4')])
    arr['id'] = obj_id
    arr['mcal_step'] = mcal_step

    msk = (
        (arr['mcal_sx_row'] >= buffer_size) &
        (arr['mcal_sx_row'] < image_size - buffer_size) &
        (arr['mcal_sx_col'] >= buffer_size) &
        (arr['mcal_sx_col'] < image_size - buffer_size))
    arr['slice_flags'][~msk] = 1

    row = arr['sx_row'] + orig_start_row + position_offset
    col = arr['sx_col'] + orig_start_col + position_offset
    ra, dec = wcs.image2sky(x=col, y=row)
    arr['ra'] = ra
    arr['dec'] = dec

    row = arr['mcal_sx_row'] + orig_start_row + position_offset
    col = arr['mcal_sx_col'] + orig_start_col + position_offset
    ra, dec = wcs.image2sky(x=col, y=row)
    arr['mcal_ra'] = ra
    arr['mcal_dec'] = dec

    return arr


def _post_process_results(
        *, outputs, obj_data, image_info, buffer_size, image_size):
    # post process results
    wcs = eu.wcsutil.WCS(
        json.loads(image_info['wcs'][obj_data['file_id'][0, 0]]))
    position_offset = image_info['position_offset'][obj_data['file_id'][0, 0]]

    output = []
    dt = 0
    for res, i, _dt in outputs:
        dt += _dt
        for mcal_step, data in res.items():
            if data.size > 0:
                wcs = eu.wcsutil.WCS(
                    json.loads(image_info['wcs'][obj_data['file_id'][i, 0]]))
                position_offset \
                    = image_info['position_offset'][obj_data['file_id'][i, 0]]

                output.append(_make_output_array(
                    data=data,
                    obj_id=obj_data['id'][i],
                    mcal_step=mcal_step,
                    orig_start_col=obj_data['orig_start_col'][i, 0],
                    orig_start_row=obj_data['orig_start_row'][i, 0],
                    wcs=wcs,
                    position_offset=position_offset,
                    buffer_size=buffer_size,
                    image_size=image_size))

    # concatenate once since generally more efficient
    output = np.concatenate(output)

    return output, dt


def _do_metadetect_and_cal(
        config, mbobs, seed, i, preprocessing_function,
        psf_rec_funcs, wcs_jacobian_func, buffer_size):
    _t0 = time.time()
    rng = np.random.RandomState(seed=seed)
    if preprocessing_function is not None:
        logger.debug("preprocessing multiband obslist %d", i)
        mbobs = preprocessing_function(mbobs=mbobs, rng=rng)
    res = do_metadetect_and_cal(
        config, mbobs, rng,
        wcs_jacobian_func=wcs_jacobian_func,
        psf_rec_funcs=psf_rec_funcs,
        buffer_size=buffer_size)
    return res, i, time.time() - _t0


def _do_metadetect(config, mbobs, seed, i, preprocessing_function):
    _t0 = time.time()
    rng = np.random.RandomState(seed=seed)
    if preprocessing_function is not None:
        logger.debug("preprocessing multiband obslist %d", i)
        mbobs = preprocessing_function(mbobs=mbobs, rng=rng)
    res = do_metadetect(config, mbobs, rng)
    return res, i, time.time() - _t0


def _get_part_ranges(part, n_parts, size):
    n_per = size // n_parts
    n_extra = size - n_per * n_parts
    n_per = np.ones(n_parts, dtype=np.int64) * n_per
    if n_extra > 0:
        n_per[:n_extra] += 1
    stop = np.cumsum(n_per)
    start = stop - n_per
    return start[part-1], n_per[part-1]


def _make_meds_iterator(mbmeds, start, num, return_local_psf_and_wcs=False):
    """This function returns a function which is used as an iterator.

    Closure closure blah blah blah.

    TLDR: Doing things this way allows the code to only read a subset of the
    images from disk in a pipelined manner.

    This works because all of the list-like things fed to joblib are actually
    generators that build their values on-the-fly.
    """
    def _func():
        for i in range(start, start+num):
            mbobs = mbmeds.get_mbobs(i)
            if return_local_psf_and_wcs:
                psf_rec_funcs = mbmeds.get_psf_rec_funcs(i)
                wcs_jacobian_func = mbmeds.get_wcs_jacobian_func(i)

                # print("FIXME: setting psf rec funcs to None", flush=True)
                psf_rec_funcs = None

                # print("FIXME: setting wcs jacobian func to None", flush=True)
                wcs_jacobian_func = None

                yield i, mbobs, psf_rec_funcs, wcs_jacobian_func
            else:
                yield i, mbobs

    return _func


def run_metadetect(
        *,
        config,
        multiband_meds,
        output_file,
        buffer_size,
        seed,
        part=1,
        n_parts=1,
        preprocessing_function=None,
        use_local_psf_and_wcs=True):
    """Run metadetect on a "pizza slice" MEDS file and write the outputs to
    disk.

    Parameters
    ----------
    config : dict
        The metadetect configuration file.
    multiband_meds : `ngmix.medsreaders.MultiBandNGMixMEDS`
        A multiband MEDS data structure.
    output_file : str
        The file to which to write the outputs.
    buffer_size : int
        The size of the buffer region in coadd pixels.
    seed : int
        The random seed to use for the run.
    part : int, optional
        The part of the file to process. Starts at 1 and runs to n_parts.
    n_parts : int, optional
        The number of parts to split the file into.
    preprocessing_function : function, optional
        An optional function to preprocessing the multiband observation
        lists before running metadetect. The function signature should
        be:
            ```
            def func(*, mbobs, rng):
                ...
                return new_mbobs
            ```
        The default of `None` does no preprocessing.
    use_local_psf_and_wcs : bool, optional
        The default of `True` will use the local PSF and WCS Jacobian at each
        detection for shear measurements. If `False`, the PSF and Jacobian used
        for the metadetection step will be used at each detection for shear
        measurements.
    """
    t0 = time.time()

    # process each slice in a pipeline
    start, num = _get_part_ranges(part, n_parts, multiband_meds.size)
    print('# of slices: %d' % num, flush=True)
    print('slice range: [%d, %d)' % (start, start+num), flush=True)
    if use_local_psf_and_wcs:
        print('using local PSF and WCS jacobian', flush=True)
        meds_iter = _make_meds_iterator(
            multiband_meds, start, num,
            return_local_psf_and_wcs=use_local_psf_and_wcs)
        outputs = joblib.Parallel(
                verbose=10,
                n_jobs=int(os.environ.get('OMP_NUM_THREADS', 1)),
                pre_dispatch='2*n_jobs',
                max_nbytes=None)(
                    joblib.delayed(_do_metadetect_and_cal)(
                        config, mbobs, seed+i*256, i,
                        preprocessing_function,
                        psf_rec_funcs,
                        wcs_jacobian_func,
                        buffer_size)
                    for i, mbobs, psf_rec_funcs, wcs_jacobian_func
                    in meds_iter())

    else:
        meds_iter = _make_meds_iterator(multiband_meds, start, num)
        outputs = joblib.Parallel(
                verbose=10,
                n_jobs=int(os.environ.get('OMP_NUM_THREADS', 1)),
                pre_dispatch='2*n_jobs',
                max_nbytes=None)(
                    joblib.delayed(_do_metadetect)(
                        config, mbobs, seed+i*256, i, preprocessing_function)
                    for i, mbobs in meds_iter())

    # join all the outputs
    output, cpu_time = _post_process_results(
        outputs=outputs,
        obj_data=multiband_meds.mlist[0].get_cat(),
        image_info=multiband_meds.mlist[0].get_image_info(),
        buffer_size=buffer_size,
        image_size=multiband_meds.get_mbobs(0)[0][0].image.shape[0])

    # report and do i/o
    wall_time = time.time() - t0
    print(
        "run time: ",
        str(datetime.timedelta(seconds=int(wall_time))),
        flush=True)
    print(
        "CPU seconds per slice: ",
        cpu_time / len(outputs), flush=True)

    fitsio.write(output_file, output, clobber=True)
