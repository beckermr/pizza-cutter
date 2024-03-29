import os
import numpy as np
import pytest

import galsim
import piff
from meds.bounds import Bounds

from .._affine_wcs import AffineWCS
from .._se_image import SEImageSlice, clear_image_and_wcs_caches


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_resample_smoke(se_image_data, coadd_image_data):
    psf_mod = piff.PSF.read(se_image_data['source_info']['piff_path'])
    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=psf_mod,
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs={"GI_COLOR": 0.61},
        noise_seeds=[10, 11, 23, 45],
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
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
    x = int(np.floor(x+0.5))
    y = int(np.floor(y+0.5))
    resampled_data = se_im.resample(
        wcs=coadd_image_data['eu_wcs'],
        wcs_position_offset=coadd_image_data['position_offset'],
        wcs_interp_shape=(10000, 10000),
        x_start=x-half,
        y_start=y-half,
        box_size=dim,
        psf_x_start=x-11,
        psf_y_start=y-11,
        psf_box_size=23,
        se_wcs_interp_delta=8,
        coadd_wcs_interp_delta=100,
    )

    # we are simply looking for weird outputs here to make sure it actually
    # runs in a simple, known case
    for k in resampled_data:
        if k not in ['noises']:
            assert np.all(np.isfinite(resampled_data[k])), k

    for ns in resampled_data['noises']:
        assert np.all(np.isfinite(ns)), 'noises'


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('eps_x', [-3, 0, 3])
@pytest.mark.parametrize('eps_y', [-5, 0, 5])
@pytest.mark.parametrize('noise_seeds', [[10], [11, 10]])
def test_se_image_resample_shifts(se_image_data, eps_x, eps_y, noise_seeds):
    # we reset the private methods on this class but our caches rely on the
    # filenames and source info, so we clear them by hand
    clear_image_and_wcs_caches()

    # SE WCS defs
    # starts at x_start, y_start in image coords
    # applies eps_y to shift where in the underlying image
    # the interp is done
    x_start = 10
    y_start = 20

    def _se_sky2image(ra, dec, find=None):
        return ra + x_start + 1, dec + y_start + 1

    def _se_image2sky(x, y):
        # SE image location of (x_start, y_start) should map to
        # (0, 0) in the sky
        return x - x_start - 1, y - y_start - 1

    # coadd WCS defs
    # starts at (coadd_x_start, coadd_y_start) in image coords
    # applies eps_x to shift location in underlying image
    # has a position offset of 4
    pos_off = 4
    coadd_x_start = 200
    coadd_y_start = 150

    class FakeWCS(AffineWCS):
        def __init__(self):
            self.dudx = None
            self.dudy = None
            self.dvdx = None
            self.dvdy = None
            self.x0 = None
            self.y0 = None

        def image2sky(self, x, y):
            # coadd position of
            # (coadd_x_start + pos_off, coadd_y_start + pos_off) should map to
            # (0, 0) in the sky
            return (
                x - pos_off - coadd_x_start,
                y - pos_off - coadd_y_start)

        def sky2image(self, longitude, latitude):
            return (
                longitude + pos_off + coadd_x_start,
                latitude + pos_off + coadd_y_start)

    class DummyWCS(AffineWCS):
        def __init__(self):
            self.dudx = None
            self.dudy = None
            self.dvdx = None
            self.dvdy = None
            self.x0 = None
            self.y0 = None

        def get_jacobian(self, x, y):
            raise NotImplementedError()

        def image2sky(self, x, y):
            raise NotImplementedError()

        def sky2image(self, longitude, latitude):
            raise NotImplementedError()

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=galsim.Gaussian(fwhm=0.8),
        wcs=se_image_data['eu_wcs'],
        wcs_position_offset=1,
        wcs_color=0,
        psf_kwargs=None,
        noise_seeds=noise_seeds,
        mask_tape_bumps=False,
        mask_piff_failure_config=None,
    )
    se_im._im_shape = (512, 512)

    # we are going to override these methods for testing
    se_im._wcs = DummyWCS()
    se_im._wcs.sky2image = _se_sky2image
    se_im._wcs.image2sky = _se_image2sky

    # now make it seem as if the image data has been read in
    # by setting it too
    box_size = 150
    half_box_size = 75
    rng = np.random.RandomState(seed=42)
    se_im.image = rng.normal(size=(box_size, box_size)) + 100
    se_im.interp_only_image = rng.normal(size=(box_size, box_size)) + 100
    se_im.interp_frac = rng.normal(size=(box_size, box_size)) + 100
    se_im.noises = [
        rng.normal(size=(box_size, box_size)) + 100
        for _ in noise_seeds
    ]
    se_im.bmask = (rng.normal(size=(box_size, box_size)) * 100).astype(np.int32)
    se_im.x_start = x_start
    se_im.y_start = y_start
    se_im.box_size = box_size
    se_im.pmask = (rng.normal(size=(box_size, box_size)) * 100).astype(np.int32)

    # fake the PSF
    se_im.psf = rng.normal(size=(55, 55)) + 100
    se_im.psf_box_size = 55
    se_im.psf_x_start = x_start + half_box_size - 27
    se_im.psf_y_start = y_start + half_box_size - 27

    # the inputs here are zero-indexed, thus they lack pos_off
    # we apply the shifts eps_x, eps_y to test moving the coadd around
    # relative to the SE image to make sure we are getting the right pixels
    # psf is close to centered around the middle of the 100x100 picel patch
    resampled_data = se_im.resample(
        wcs=FakeWCS(),
        wcs_position_offset=pos_off,
        wcs_interp_shape=(2000, 2000),
        x_start=coadd_x_start + half_box_size - 50 + eps_x,
        y_start=coadd_y_start + half_box_size - 50 + eps_y,
        box_size=100,
        psf_x_start=coadd_x_start + half_box_size - 11 + eps_x,
        psf_y_start=coadd_y_start + half_box_size - 11 + eps_y,
        psf_box_size=23,
        se_wcs_interp_delta=8,
        coadd_wcs_interp_delta=10,
    )

    # first check they are finite
    for k in resampled_data:
        if k != 'bmask' and k != 'ormask' and k != 'pmask':
            assert np.all(np.isfinite(resampled_data[k])), k

    # now check the values
    # whole pixel shifts interpolate exactly, up to numerical errors
    # for eps_x, eps_y = (0, 0), the coadd is located in the central 100 x 100
    # pixels of the SE image
    wcs = FakeWCS()
    ra, dec = wcs.image2sky(
        # using pos_off here since calling directly!
        coadd_x_start + pos_off + half_box_size - 50 + eps_x,
        coadd_y_start + pos_off + half_box_size - 50 + eps_y)
    final_x_start, final_y_start = _se_sky2image(ra, dec)
    final_x_start -= 1
    final_y_start -= 1
    final_x_start -= x_start
    final_y_start -= y_start
    for k in ['image', 'bmask', 'pmask', 'interp_only_image', 'interp_frac']:
        assert np.allclose(
            resampled_data[k],
            getattr(se_im, k)[final_y_start:final_y_start+100,
                              final_x_start:final_x_start+100]), k

    for i in range(len(noise_seeds)):
        assert np.allclose(
            resampled_data["noises"][i],
            se_im.noises[i][
                final_y_start:final_y_start+100,
                final_x_start:final_x_start+100
            ],
        ), i

    ra, dec = wcs.image2sky(
        # using pos_off here since calling directly!
        # need extra offset for the PSF stamp
        coadd_x_start + pos_off + half_box_size - 11 + eps_x,
        coadd_y_start + pos_off + half_box_size - 11 + eps_y)
    final_x_start, final_y_start = _se_sky2image(ra, dec)
    final_x_start -= 1
    final_y_start -= 1
    final_x_start -= (x_start + half_box_size - 27)
    final_y_start -= (y_start + half_box_size - 27)

    true_psf = se_im.psf[
        final_y_start:final_y_start+23,
        final_x_start:final_x_start+23
    ].copy()
    true_psf /= np.sum(true_psf)
    assert np.allclose(
        resampled_data['psf'],
        true_psf,
    )
