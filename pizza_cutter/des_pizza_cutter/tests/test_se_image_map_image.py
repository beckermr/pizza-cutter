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
@pytest.mark.parametrize("coadd_dim", [6, 10, 20])
def test_se_image_map_image(se_image_data, coadd_image_data, coadd_dim):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        noise_seeds=[10, 11, 23, 45],
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
    se_im.set_psf(ra, dec)
    se_im.set_interp_image_noise_pmask(
        interp_image=se_im.image,
        interp_noises=se_im.noises,
        mask=np.zeros_like(se_im.bmask),
    )
    x, y = coadd_image_data['eu_wcs'].sky2image(ra, dec)
    x = int(np.floor(x+0.5)) - coadd_image_data['position_offset']
    y = int(np.floor(y+0.5)) - coadd_image_data['position_offset']
    x_start = x - coadd_dim//2
    y_start = y - coadd_dim//2
    coadd_img = np.arange(coadd_dim**2).reshape((coadd_dim, coadd_dim))
    res = se_im.map_image_by_nearest_pixel(
        image=coadd_img,
        wcs=coadd_image_data['eu_wcs'],
        wcs_position_offset=coadd_image_data['position_offset'],
        x_start=x_start,
        y_start=y_start,
        wcs_interp_delta=8,
    )
    assert np.all(np.isfinite(res))

    for row in range(res.shape[0]):
        for col in range(res.shape[1]):
            y = row + se_im.y_start
            x = col + se_im.x_start
            ra, dec = se_im.image2sky(x, y)
            xc, yc = coadd_image_data['eu_wcs'].sky2image(ra, dec)
            xc -= coadd_image_data['position_offset']
            yc -= coadd_image_data['position_offset']
            xc -= x_start
            yc -= y_start
            xc = int(np.floor(xc+0.5))
            yc = int(np.floor(yc+0.5))
            if yc >= 0 and yc < coadd_dim and xc >= 0 and xc < coadd_dim:
                assert coadd_img[yc, xc] == res[row, col]
            else:
                assert 0 == res[row, col]
