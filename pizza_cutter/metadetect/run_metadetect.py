import time
import numpy as np

from metadetect.metadetect import do_metadetect
from ngmix.medsreaders import MultiBandMEDS, MEDS


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
        MEDS(fname) for fname in meds_file_list]
    mbmeds = MultiBandMEDS(meds_list)

    for i in range(mbmeds.size):
        mbobs = mbmeds.get_mbobs(i)
        res = do_metadetect(config, mbobs, rng)
        print(res)
        break
    total_time = time.time() - t0
    print("time per: ", total_time / 1)
