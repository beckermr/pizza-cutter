import os
import numpy as np
import pytest

import piff
import galsim

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
    resampled_data = se_im.resample(
        wcs=coadd_image_data['eu_wcs'],
        wcs_position_offset=coadd_image_data['position_offset'],
        x_start=600-250,
        y_start=700-250,
        box_size=500,
        psf_x_start=600-23,
        psf_y_start=700-23,
        psf_box_size=23
    )


@pytest.mark.parametrize('eps_x', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize('eps_y', [
    -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
def test_se_image_resample_shifts(se_image_data, eps_x, eps_y):

    def _sky2image(ra, dec):
        return ra, dec + eps_y

    def _image2sky(x, y):
        return x + eps_x, y

    assert False
