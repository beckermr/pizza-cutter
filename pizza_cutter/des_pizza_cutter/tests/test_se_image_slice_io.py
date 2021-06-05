import os
import numpy as np
import pytest
import tempfile

import fitsio
from meds.bounds import Bounds

from .._se_image import SEImageSlice
from .._constants import MAGZP_REF
from ...memmappednoise import MemMappedNoiseImage


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice i/o can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_slice_read(monkeypatch, se_image_data):

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv(
            'DESDATA', os.path.join(os.environ['TEST_DESDATA'], 'DESDATA'))

        se_im = SEImageSlice(
            source_info=se_image_data['source_info'],
            psf_model=None,
            wcs=se_image_data['eu_wcs'],
            wcs_position_offset=1,
            wcs_color=0,
            noise_seed=10,
            mask_tape_bumps=False,
            tmpdir=tmpdir,
        )
        patch_bnds = Bounds(
            rowmin=50,
            rowmax=50+32-1,
            colmin=10,
            colmax=10+32-1,
        )
        se_im.set_slice(patch_bnds)

        # check the attributes
        assert se_im.x_start == 10
        assert se_im.y_start == 50
        assert se_im.box_size == 32

        # check the images
        im = fitsio.read(
            se_image_data['source_info']['image_path'],
            ext=se_image_data['source_info']['image_ext'])
        bkg = fitsio.read(
            se_image_data['source_info']['bkg_path'],
            ext=se_image_data['source_info']['bkg_ext'])
        im -= bkg
        scale = 10.0**(0.4*(MAGZP_REF - se_image_data['source_info']['magzp']))
        im *= scale
        assert np.array_equal(im[50:82, 10:42], se_im.image)

        wgt = fitsio.read(
            se_image_data['source_info']['weight_path'],
            ext=se_image_data['source_info']['weight_ext'])
        wgt /= scale**2
        assert np.array_equal(wgt[50:82, 10:42], se_im.weight)

        bmask = fitsio.read(
            se_image_data['source_info']['bmask_path'],
            ext=se_image_data['source_info']['bmask_ext'])
        assert np.array_equal(bmask[50:82, 10:42], se_im.bmask)

        zmsk = wgt <= 0
        nse = MemMappedNoiseImage(
            seed=10,
            weight=wgt * (~zmsk) + zmsk * np.max(wgt[~zmsk]),
            dir=tmpdir,
            sx=1024,
            sy=1024,
        )
        assert np.array_equal(nse[50:82, 10:42], se_im.noise)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice i/o can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_slice_noise_adjacent(monkeypatch, se_image_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv(
            'DESDATA', os.path.join(os.environ['TEST_DESDATA'], 'DESDATA'))

        se_im = SEImageSlice(
            source_info=se_image_data['source_info'],
            psf_model=None,
            wcs=se_image_data['eu_wcs'],
            wcs_position_offset=1,
            wcs_color=0,
            noise_seed=10,
            mask_tape_bumps=False,
            tmpdir=tmpdir,
        )
        patch_bnds = Bounds(
            rowmin=50,
            rowmax=50+30-1,
            colmin=10,
            colmax=10+30-1,
        )
        se_im.set_slice(patch_bnds)

        se_im_adj = SEImageSlice(
            source_info=se_image_data['source_info'],
            psf_model=None,
            wcs=se_image_data['eu_wcs'],
            wcs_position_offset=1,
            wcs_color=0,
            noise_seed=10,
            mask_tape_bumps=False,
            tmpdir=tmpdir,
        )
        patch_bnds = Bounds(
            rowmin=50,
            rowmax=50+30-1,
            colmin=20,
            colmax=20+30-1,
        )
        se_im_adj.set_slice(patch_bnds)

        # make sure the overlapping parts of the noise field are the same
        assert np.array_equal(se_im.noise[:, 10:], se_im_adj.noise[:, :-10])


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice i/o can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_slice_double_use(monkeypatch, se_image_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv(
            'DESDATA', os.path.join(os.environ['TEST_DESDATA'], 'DESDATA'))

        se_im = SEImageSlice(
            source_info=se_image_data['source_info'],
            psf_model=None,
            wcs=se_image_data['eu_wcs'],
            wcs_position_offset=1,
            wcs_color=0,
            noise_seed=10,
            mask_tape_bumps=False,
            tmpdir=tmpdir,
        )
        patch_bnds = Bounds(
            rowmin=50,
            rowmax=50+32-1,
            colmin=10,
            colmax=10+32-1,
        )
        se_im.set_slice(patch_bnds)

        assert se_im.x_start == 10
        assert se_im.y_start == 50
        assert se_im.box_size == 32

        im = fitsio.read(
            se_image_data['source_info']['image_path'],
            ext=se_image_data['source_info']['image_ext'])
        bkg = fitsio.read(
            se_image_data['source_info']['bkg_path'],
            ext=se_image_data['source_info']['bkg_ext'])
        im -= bkg
        scale = 10.0**(0.4*(MAGZP_REF - se_image_data['source_info']['magzp']))
        im *= scale
        assert np.array_equal(im[50:82, 10:42], se_im.image)

        # now we move to another spot
        patch_bnds = Bounds(
            rowmin=50,
            rowmax=50+32-1,
            colmin=20,
            colmax=20+32-1,
        )
        se_im.set_slice(patch_bnds)

        assert se_im.x_start == 20
        assert se_im.y_start == 50
        assert se_im.box_size == 32

        im = fitsio.read(
            se_image_data['source_info']['image_path'],
            ext=se_image_data['source_info']['image_ext'])
        bkg = fitsio.read(
            se_image_data['source_info']['bkg_path'],
            ext=se_image_data['source_info']['bkg_ext'])
        im -= bkg
        scale = 10.0**(0.4*(MAGZP_REF - se_image_data['source_info']['magzp']))
        im *= scale
        assert np.array_equal(im[50:82, 20:52], se_im.image)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice i/o can only be tested if '
        'test data is at TEST_DESDATA'))
def test_se_image_slice_two_obj(monkeypatch, se_image_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv(
            'DESDATA', os.path.join(os.environ['TEST_DESDATA'], 'DESDATA'))

        se_im = SEImageSlice(
            source_info=se_image_data['source_info'],
            psf_model=None,
            wcs=se_image_data['eu_wcs'],
            wcs_position_offset=1,
            wcs_color=0,
            noise_seed=10,
            mask_tape_bumps=False,
            tmpdir=tmpdir,
        )
        patch_bnds = Bounds(
            rowmin=50,
            rowmax=50+32-1,
            colmin=10,
            colmax=10+32-1,
        )
        se_im.set_slice(patch_bnds)

        assert se_im.x_start == 10
        assert se_im.y_start == 50
        assert se_im.box_size == 32

        im = fitsio.read(
            se_image_data['source_info']['image_path'],
            ext=se_image_data['source_info']['image_ext'])
        bkg = fitsio.read(
            se_image_data['source_info']['bkg_path'],
            ext=se_image_data['source_info']['bkg_ext'])
        im -= bkg
        scale = 10.0**(0.4*(MAGZP_REF - se_image_data['source_info']['magzp']))
        im *= scale
        assert np.array_equal(im[50:82, 10:42], se_im.image)

        # now we move to another spot
        se_im1 = SEImageSlice(
            source_info=se_image_data['source_info'],
            psf_model=None,
            wcs=se_image_data['eu_wcs'],
            wcs_position_offset=1,
            wcs_color=0,
            noise_seed=10,
            mask_tape_bumps=False,
            tmpdir=tmpdir,
        )
        patch_bnds = Bounds(
            rowmin=50,
            rowmax=50+32-1,
            colmin=20,
            colmax=20+32-1,
        )
        se_im1.set_slice(patch_bnds)

        assert se_im1.x_start == 20
        assert se_im1.y_start == 50
        assert se_im1.box_size == 32

        im = fitsio.read(
            se_image_data['source_info']['image_path'],
            ext=se_image_data['source_info']['image_ext'])
        bkg = fitsio.read(
            se_image_data['source_info']['bkg_path'],
            ext=se_image_data['source_info']['bkg_ext'])
        im -= bkg
        scale = 10.0**(0.4*(MAGZP_REF - se_image_data['source_info']['magzp']))
        im *= scale
        assert np.array_equal(im[50:82, 20:52], se_im1.image)
