import os
import json
import time
import datetime
import joblib

import numpy as np
import esutil as eu
import fitsio

from metadetect.metadetect import do_metadetect
from ngmix.medsreaders import MultiBandNGMixMEDS, NGMixMEDS


def _make_output_array(
        *,
        data, obj_id, mcal_step,
        orig_start_row, orig_start_col, position_offset, wcs):
    arr = eu.numpy_util.add_fields(
                data,
                [('id', 'i8'), ('mcal_step', 'S7'),
                 ('ra', 'f8'), ('dec', 'f8')])
    arr['id'] = obj_id
    arr['mcal_step'] = mcal_step

    row = arr['sx_row'] + orig_start_row + position_offset
    col = arr['sx_col'] + orig_start_col + position_offset
    ra, dec = wcs.image2sky(x=col, y=row)
    arr['ra'] = ra
    arr['dec'] = dec

    return arr


def _post_process_results(*, outputs, obj_data, image_info):
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
                output.append(_make_output_array(
                    data=data,
                    obj_id=obj_data['id'][i],
                    mcal_step=mcal_step,
                    orig_start_col=obj_data['orig_start_col'][i, 0],
                    orig_start_row=obj_data['orig_start_row'][i, 0],
                    wcs=wcs,
                    position_offset=position_offset))

    # concatenate once since generally more efficient
    output = np.concatenate(output)

    return output, dt


def _make_output_filename(meds_fname):
    fname = os.path.basename(meds_fname)
    fname = fname.replace('.fz', '').replace('.fits', '')
    # remove the band - always comes after the tilename by convention
    items = fname.split('_')
    fname = '_'.join(items[0:1] + items[2:])
    return fname + '-metadetect-output.fits'


def _do_metadetect(config, mbobs, seed, i):
    _t0 = time.time()
    rng = np.random.RandomState(seed=seed)
    res = do_metadetect(config, mbobs, rng)
    return res, i, time.time() - _t0


def _make_meds_iterator(mbmeds):
    """This function returns a function which is used as an iterator.

    Closure closure blah blah blah.

    TLDR: Doing things this way allows the code to only read a subset of the
    images from disk in a pipelined manner.

    This works because all of the list-like things fed to joblib are actually
    generators that build their values on-the-fly.
    """
    def _func():
        for i in range(mbmeds.size):
            mbobs = mbmeds.get_mbobs(i)
            yield i, mbobs
    return _func


def run_metadetect(
        *,
        config,
        meds_file_list,
        output_path,
        seed):
    """Run metadetect on a "pizza slice" MEDS file and write the outputs to
    disk.

    Parameters
    ----------
    config : dict
        The metadetect configuration file.
    meds_file_list : list of str
        A list of the input MEDS files to run on.
    output_path : str
        The path to which to write the outputs.
    """
    t0 = time.time()

    # init meds once here
    meds_list = [NGMixMEDS(fname) for fname in meds_file_list]
    mbmeds = MultiBandNGMixMEDS(meds_list)

    # process each slice in a pipeline
    meds_iter = _make_meds_iterator(mbmeds)
    outputs = joblib.Parallel(
            verbose=10,
            n_jobs=int(os.environ.get('OMP_NUM_THREADS', 1)),
            pre_dispatch='2*n_jobs')(
        joblib.delayed(_do_metadetect)(config, mbobs, seed+i, i)
        for i, mbobs in meds_iter())

    # join all the outputs
    output, cpu_time = _post_process_results(
        outputs=outputs,
        obj_data=meds_list[0].get_cat(),
        image_info=meds_list[0].get_image_info())

    # report and do i/o
    wall_time = time.time() - t0
    print(
        "run time: ",
        str(datetime.timedelta(seconds=int(wall_time))),
        flush=True)
    print(
        "CPU seconds per slice: ",
        cpu_time / len(outputs), flush=True)

    fname = os.path.join(
        output_path,
        _make_output_filename(meds_file_list[0]))
    print("output file:", fname, flush=True)
    fitsio.write(fname, output, clobber=True)
