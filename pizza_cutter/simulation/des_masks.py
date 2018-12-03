import subprocess
import logging

import fitsio
import numpy as np
import blosc

from ..des_pizza_cutter._slice_flagging import compute_unmasked_trail_fraction

logger = logging.getLogger(__name__)


def gen_masks_from_des_y3_images(
        *, bmask_paths, bmask_ext, output_file, n_per, seed, nrows, ncols):
    """Generate a set of bit masks from DES Y3 bit masks for image simulations.

    Parameters
    ----------
    bmask_paths : list of str
        A list of file paths to load bit masks from for randomly sampling.
    bmask_ext : int or str
        The FITS extension of the bit mask in each file.
    output_file : str
        The path to output the bit mask library FITS file.
    n_per : int
        The number of randomly chosen patches of each bit mask from
        `bmask_paths` to attempt to keep.
    seed : int
        The random seed for selecting patches.
    nrows : int
        The number of rows of each bit mask patch.
    ncols : int
        The number of columns of each bit mask patch.

    Notes
    -----
    The number of rows and columns of each patch must be less than or equal to
    half of the corresponding input bit mask image dimensions.
    """
    # these regions in the input image have all of their bits ignored
    # 32 is a star mask so that we ignore all of the crap around stars
    bad_region_flag = 32

    # these bits are ignored everywhere
    # we ignore stars (32) and streaks (1024)
    flags_to_ignore = 32 | 1024

    # we ignore whole masks where star bleeds (64) are not
    # also masked by bit 32
    max_unmasked_trail_fraction = 0.0
    max_masked_frac = 0.1

    # good defaults since we will pack them as a single 1-d image
    fpack_pars = {
        'FZQVALUE': 4,
        'FZTILE': "(10240,1)",
        'FZALGOR': "RICE_1",
        # preserve zeros, don't dither them
        'FZQMETHD': "SUBTRACTIVE_DITHER_2",
    }

    rng = np.random.RandomState(seed=seed)

    # we don't know a priori how many masks we can generate.
    # we try a fixed number of times per input mask to avoid oversampling
    # regions where a lot of the full mask is excluded.
    # thus we will store the final masks in memory until we write them out.
    # I am using blosc to compress them in memory.
    final_masks = []
    for bmask_path in sorted(bmask_paths):
        logger.info('processing bmask: %s', bmask_path)
        with fitsio.FITS(bmask_path) as fits:
            bmask = fits[bmask_ext]
            for _ in range(n_per):
                msk = _gen_bmask_from_image(
                    bmask=bmask,
                    nrows=nrows,
                    ncols=ncols,
                    flags_to_ignore=flags_to_ignore,
                    bad_region_flag=bad_region_flag,
                    random_state=rng)

                msk_frac = np.mean(msk != 0)

                if (compute_unmasked_trail_fraction(bmask=msk) <=
                        max_unmasked_trail_fraction) and (
                            msk_frac < max_masked_frac):
                    final_masks.append(blosc.pack_array(msk))

    # finally we write them to disk and compress
    logger.info('writing the data')
    msk_size = nrows * ncols
    dims = (msk_size * len(final_masks), )
    logger.info('image uses %f GB', dims[0] * 4 / 1000 / 1000 / 1000)
    with fitsio.FITS(output_file, clobber=True, mode='rw') as fits:
        # create the header and make sure to write the fpack keys so that
        # fpack does the right thing
        fits.create_image_hdu(
            img=None,
            dtype='i4',
            dims=dims,
            extname='msk',
            header=fpack_pars)
        fits[0].write_keys(fpack_pars, clean=False)

        start_row = 0
        for msk in final_masks:
            msk = blosc.unpack_array(msk)
            fits[0].write(msk.ravel(), start=start_row)
            start_row += msk_size
        assert start_row == dims[0]

        # store the metadata we need to get the images back
        metadata = np.zeros(1, dtype=[('nrows', 'i4'), ('ncols', 'i4')])
        metadata['nrows'] = nrows
        metadata['ncols'] = ncols
        fits.write(metadata, extname='metadata')

    # now we fpack - this will save a ton of space because fpacking
    # integers is very efficient
    logger.info('fpacking the outputs')
    try:
        cmd = 'fpack ' + output_file
        subprocess.check_call(cmd, shell=True)
    except Exception:
        pass


def _gen_bmask_from_image(
        *, bmask, nrows, ncols, flags_to_ignore,
        bad_region_flag, random_state):
    """Generate bit masks from an input bitmask by randomly sampling patches
    of it.

    Parameters
    ----------
    bmask : np.array or `fitsio.ImageHDU`
        An input image to draw bmasks from.
    nrows : int
        The number of rows (y-axis, first index) of the patch.
    ncols : int
        The number of columns (x-axis, second index) of the patch.
    flags_to_ignore : int
        Any flags that should be zeroed in the output.
    bad_region_flag : int
        A flag bit for any region where all flags should be ignored.
    random_state : `np.random.RandomState`
        An RNG.

    Returns
    -------
    msk : np.array
        A random bit mask from the image.
    """

    # this will handle fitsio HDUs as well
    try:
        dims = bmask.get_dims()
    except AttributeError:
        dims = bmask.shape

    # usually I am not for this kind of check, but it seems pertinent here.
    if nrows > dims[0] // 2 or ncols > dims[1] // 2:
        raise ValueError(
            "The input image is too small to select random pacthes!")

    # draw the lower left corner
    start_row = random_state.randint(0, dims[0] - nrows)
    start_col = random_state.randint(0, dims[1] - ncols)

    assert start_row + nrows <= dims[0]
    assert start_col + ncols <= dims[1]

    # read mask and set any flags we don't need to zero
    # we make a copy to make sure we are not changing the input data
    msk = bmask[start_row:start_row+nrows, start_col:start_col+ncols].copy()
    not_flags_to_ignore = ~flags_to_ignore
    bad_region_msk = (msk & bad_region_flag) != 0

    msk &= not_flags_to_ignore
    msk[bad_region_msk] = 0

    return msk
