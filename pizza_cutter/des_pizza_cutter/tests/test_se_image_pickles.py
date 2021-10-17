import os
import pickle
import pytest

import numpy as np

import piff

from .._se_image import SEImageSlice


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('noise_seeds', [[10], [1232, 25, 22]])
def test_se_image_pickles(se_image_data, noise_seeds):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs={"GI_COLOR": 0.61},
        noise_seeds=noise_seeds,
        mask_tape_bumps=False,
    )

    psf_im = se_im.get_psf_image(20, 20.5)
    ra, dec = se_im.image2sky(10, 10)

    se_pickle = pickle.dumps(se_im)
    se_im_new = pickle.loads(se_pickle)

    psf_im_new = se_im_new.get_psf_image(20, 20.5)
    ra_new, dec_new = se_im_new.image2sky(10, 10)
    assert np.array_equal(psf_im, psf_im_new)
    assert np.array_equal(dec, dec_new)
    assert np.array_equal(ra, ra_new)
