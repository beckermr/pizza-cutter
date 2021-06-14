import os
import numpy as np
import pytest

import piff
from meds.bounds import Bounds

from .._se_image import SEImageSlice


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_interp_set(se_image_data, coadd_image_data):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        noise_seeds=[10, 12, 11],
        mask_tape_bumps=False,
    )
    se_im._im_shape = (512, 512)
    dim = 10
    half = 5
    ra, dec = se_im.image2sky(300, 150)
    patch_bnds = Bounds(
        rowmin=150-half,
        rowmax=150-half+dim-1,
        colmin=300-half,
        colmax=300-half+dim-1,
    )
    se_im.set_slice(patch_bnds)

    orig_image = se_im.image.copy()
    orig_noises = [nse.copy() for nse in se_im.noises]

    pmask = np.zeros_like(se_im.image, dtype=np.int32)
    pmask[1, 2] = 32
    msk = pmask != 0
    interp_image = orig_image.copy()
    interp_noises = [nse.copy() for nse in orig_noises]
    interp_image[msk] = 42.12345
    for i in range(len(interp_noises)):
        interp_noises[i][msk] = 42.12345
    interp_frac = np.zeros_like(se_im.image)
    interp_frac[msk] = 1

    interp_only_image = interp_image.copy()
    interp_only_image[~msk] = 0

    se_im.set_interp_image_noise_pmask(
        interp_image=interp_image,
        interp_noises=interp_noises,
        mask=pmask,
    )

    assert np.array_equal(se_im.pmask, pmask)
    assert np.array_equal(se_im.orig_image, orig_image)
    assert np.array_equal(se_im.image, interp_image)
    for i in range(len(interp_noises)):
        assert np.array_equal(se_im.noises[i], interp_noises[i])
    assert np.array_equal(se_im.interp_frac, interp_frac)
    assert np.array_equal(se_im.interp_only_image, interp_only_image)
