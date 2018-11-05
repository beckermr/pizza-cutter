import os
import time
import datetime

import numpy as np

import fitsio

from metadetect.metadetect import do_metadetect
from ngmix.medsreaders import MultiBandNGMixMEDS, NGMixMEDS


def _make_output_array(*, data, obj_id, mcal_step):
    dt = data.dtype.descr + [('id', 'i8'), ('mcal_step', 'S7')]
    arr = np.zeros(data.size, dtype=dt)
    for col in data.dtype.names:
        arr[col] = data[col]
    arr['id'] = obj_id
    arr['mcal_step'] = mcal_step
    return arr


def _make_output_filename(meds_fname):
    fname = os.path.basename(meds_fname)
    fname = fname.replace('.fz', '').replace('.fits', '')
    # remove the band
    fname = '-'.join(fname.split('-')[:-1])
    return fname + '-metadetect-output.fits'


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
    rng = np.random.RandomState(seed=seed)

    t0 = time.time()
    meds_list = [
        NGMixMEDS(fname) for fname in meds_file_list]
    mbmeds = MultiBandNGMixMEDS(meds_list)

    obj_data = meds_list[0].get_cat()
    output = []
    for i in range(mbmeds.size):
        _t0 = time.time()
        print("processing slice:", obj_data['id'][i], flush=True)
        mbobs = mbmeds.get_mbobs(i)
        res = do_metadetect(config, mbobs, rng)
        for mcal_step, data in res.items():
            output.append(_make_output_array(
                data=data,
                obj_id=obj_data['id'][i],
                mcal_step=mcal_step))
        print("    time:", time.time() - _t0, flush=True)
        print(
            "    eta:",
            str(datetime.timedelta(
                seconds=int(
                    (time.time() - t0) / (i+1) * (mbmeds.size - i - 1)))),
            flush=True)

    total_time = time.time() - t0
    print("time per catalog entry: ", total_time / len(output) * 5, flush=True)

    # concatenate once since generally more efficient
    output = np.concatenate(output)
    fname = os.path.join(
        output_path,
        _make_output_filename(meds_file_list[0]))
    print("output file:", fname, flush=True)
    fitsio.write(fname, output, clobber=True)
