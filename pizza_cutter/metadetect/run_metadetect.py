

def run_metadetect(
        *,
        config,
        meds_file_list,
        output_path):
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
    print(config, meds_file_list, output_path)
