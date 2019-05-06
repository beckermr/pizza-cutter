import os
import numpy as np
import pytest

import piff

from .._se_image import SEImageSlice


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice i/o can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_resample_smoke(se_image_data, coadd_image_data):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'], noise_seed=10)
    ra, dec = se_im.image2sky(600, 700)
    se_im.set_slice(600-250, 700-250, 500)
    se_im.set_psf(ra, dec)
    x, y = coadd_image_data['eu_wcs'].sky2image(ra, dec)
    x = int(x+0.5)
    y = int(y+0.5)
    resampled_data = se_im.resample(
        wcs=coadd_image_data['eu_wcs'],
        wcs_position_offset=coadd_image_data['position_offset'],
        x_start=x-250,
        y_start=y-250,
        box_size=500,
        psf_x_start=x-11,
        psf_y_start=y-11,
        psf_box_size=23
    )

    # we are simply looking for weird outputs here to make sure it actually
    # runs in a simple, known case
    for k in resampled_data:
        if k != 'bmask' and k != 'ormask':
            assert np.all(np.isfinite(resampled_data[k])), k


# @pytest.mark.parametrize('eps_x', [
#     -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
# @pytest.mark.parametrize('eps_y', [
#     -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
# def test_se_image_resample_shifts(se_image_data, eps_x, eps_y):
#
#     def _sky2image(ra, dec):
#         return ra, dec + eps_y
#
#     def _image2sky(x, y):
#         return x + eps_x, y
#
#     assert False
