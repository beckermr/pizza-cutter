import os
import unittest.mock
import json

import numpy as np
import fitsio
import yaml
import pytest
import galsim
from esutil.wcsutil import WCS

from ..medsreader import POSITION_OFFSET, MAGZP_REF
from ..medsreader import CoaddSimSliceMEDS
from ..memmappednoise import MemMappedNoiseImage
from ..galsim_psf import GalSimPSF


class FakePSF(object):
    """A class to fake PSF measurements.
    """
    def get_rec(self, row, col):
        rng = np.random.RandomState(seed=int(col + 1000 * row))
        return rng.normal(size=(33, 33))

    def get_center(self, row, col):
        rng = np.random.RandomState(seed=int(col + 1000 * row))
        return rng.normal(size=2)

    def get_sigma(self, row, col):
        rng = np.random.RandomState(seed=int(col + 1000 * row))
        return rng.normal()


class FakeBMaskGenerator(object):
    """A class to fake the bit mask generator"""
    def get_bmask(self, index):
        bmask = np.zeros((75, 75), dtype=np.int32)
        loc = index % 10 + 5
        bmask[loc, 20:30] = 8
        bmask[20:30, loc] = 16
        return bmask

    def close(self):
        pass


@pytest.fixture
def data(tmpdir_factory):
    tmpdir = tmpdir_factory.getbasetemp()
    seed = 42
    rng = np.random.RandomState(seed=seed)
    coadd_size = 100

    wcs_header = dict(
        naxis1=100,
        naxis2=100,
        ctype1='RA---TAN',
        ctype2='DEC--TAN',
        crpix1=50.5,
        crpix2=50.5,
        cd1_1=-7.305555555556E-05,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=7.305555555556E-05,
        cunit1='deg     ',
        cunit2='deg     ',
        crval1=321.417528,
        crval2=1.444444)
    wcs = WCS(wcs_header)

    image_path = os.path.join(tmpdir, 'img.fits')
    image_ext = 0
    y, x = np.mgrid[0:coadd_size, 0:coadd_size]
    image = (10 + x*5).astype(np.float32)
    fitsio.write(image_path, image, header=wcs_header, clobber=True)

    weight_path = os.path.join(tmpdir, 'img.fits')
    weight_ext = 1
    weight = np.exp(rng.normal(size=(coadd_size, coadd_size)))
    fitsio.write(weight_path, weight)

    # we set the background to a constant to test the inteprolator
    bkg_path = os.path.join(tmpdir, 'bkg.fits')
    bkg_ext = 1
    bkg = np.ones_like(image) * 9
    fitsio.write(bkg_path, bkg, clobber=True)
    fitsio.write(bkg_path, bkg)

    seg_path = os.path.join(tmpdir, 'seg.fits')
    seg_ext = 2
    seg = (np.abs(rng.normal(size=(coadd_size, coadd_size))) * 10).astype('i4')
    fitsio.write(seg_path, seg * 0 + -1, clobber=True)
    fitsio.write(seg_path, seg * 0)
    fitsio.write(seg_path, seg)

    bmask_path = os.path.join(tmpdir, 'bmask.fits')
    bmask_ext = 0
    bmask = (np.abs(
        rng.normal(size=(coadd_size, coadd_size))) * 10).astype('i4')
    fitsio.write(bmask_path, bmask, clobber=True)

    psf_path = os.path.join(tmpdir, 'psf.psf')
    bmask_catalog = os.path.join(tmpdir, 'bmask.bmask')

    config = {
        'central_size': 25,
        'buffer_size': 25,

        'image_path': image_path,
        'image_ext': image_ext,
        'bkg_path': bkg_path,
        'bkg_ext': bkg_ext,
        'weight_path': weight_path,
        'weight_ext': weight_ext,
        'seg_path': seg_path,
        'seg_ext': seg_ext,
        'bmask_path': bmask_path,
        'bmask_ext': bmask_ext,
        'psf': psf_path,
        'seed': seed,
        'masking_config': {
            'symmetrize_masking': False,
            'se_interp_flags': 8,
            'noise_interp_flags': 16,
            'bmask_catalog': bmask_catalog}}

    return {
        'config_str': yaml.dump(config),
        'config': config,
        'noise': MemMappedNoiseImage(seed=seed, weight=weight, sx=10, sy=10),
        'image': image,
        'weight': weight,
        'bkg': bkg,
        'seg': seg,
        'bmask': bmask,
        'meds_path': os.path.join(tmpdir, 'meds.fits'),
        'wcs': wcs,
        'wcs_header': {k.upper(): v for k, v in wcs_header.items()},
        'nobj': ((100 - 25 - 25) // 25)**2}


# I am mocking out the psfex call. I HATE doing this, but the effort to
# build fake PSFEx input data is just too great to justify this test. I am only
# looking for the data to appear in the MEDS file in the right spot. This
# means the test functionally still tests the right thing.
# Ditto for the bit mask generator.
@pytest.mark.parametrize('using_psfex', [True, False])
@unittest.mock.patch('pizza_cutter.coadd_sim_slicer.medsreader.psfex.PSFEx')
@unittest.mock.patch('pizza_cutter.coadd_sim_slicer.medsreader.BMaskGenerator')
def test_medsreader_masking(bmask_mock, psf_mock, using_psfex, data):
    if using_psfex:
        pex = FakePSF()
        psf_mock.return_value = pex
    else:
        gs_conf = {'type': 'Gaussian', 'fwhm': 0.9}
        data['config']['psf'] = gs_conf
        pex = GalSimPSF(
            gs_conf,
            galsim.FitsWCS(header=data['wcs_header']))

    bmask_gen = FakeBMaskGenerator()
    bmask_mock.return_value = bmask_gen

    kwargs = dict(
        central_size=data['config']['central_size'],
        buffer_size=data['config']['buffer_size'],
        image_path=data['config']['image_path'],
        image_ext=data['config']['image_ext'],
        weight_path=data['config']['weight_path'],
        weight_ext=data['config']['weight_ext'],
        bmask_path=data['config']['bmask_path'],
        bmask_ext=data['config']['bmask_ext'],
        bkg_path=data['config']['bkg_path'],
        bkg_ext=data['config']['bkg_ext'],
        seg_path=data['config']['seg_path'],
        seg_ext=data['config']['seg_ext'],
        psf=data['config']['psf'],
        seed=data['config']['seed'],
        noise_size=10,
        masking_config=data['config']['masking_config'])

    # I am testing the non-fpacked file since fpacking makes the arrays
    # lose precision.
    with CoaddSimSliceMEDS(**kwargs) as m:
        if using_psfex:
            psf_mock.assert_called_with(data['config']['psf'])
        else:
            psf_mock.assert_not_called()

        obj = m.get_cat()
        assert len(obj) == data['nobj']

        for i in range(data['nobj']):
            assert obj['id'][i] == i
            assert obj['box_size'][i] == 75
            assert obj['ncutout'][i] == 1
            assert obj['file_id'][i, 0] == 0

            _y = i % 2
            _x = (i - i % 2) // 2
            assert i == _y + 2 * _x
            orig_col = 25 + 25 * _x + 12
            orig_row = 25 + 25 * _y + 12

            ra, dec = data['wcs'].image2sky(orig_col + 1, orig_row + 1)
            assert np.allclose(obj['ra'][i], ra)
            assert np.allclose(obj['dec'][i], dec)

            assert obj['start_row'][i, 0] == 75 * 75 * i
            assert obj['orig_col'][i, 0] == orig_col
            assert obj['orig_row'][i, 0] == orig_row
            assert obj['orig_start_col'][i, 0] == orig_col - 25 - 12
            assert obj['orig_start_row'][i, 0] == orig_row - 25 - 12
            assert obj['cutout_col'][i, 0] == 12
            assert obj['cutout_row'][i, 0] == 12

            jacob = data['wcs'].get_jacobian(orig_col + 1, orig_row + 1)
            assert obj['dudcol'][i, 0] == jacob[0]
            assert obj['dudrow'][i, 0] == jacob[1]
            assert obj['dvdcol'][i, 0] == jacob[2]
            assert obj['dvdrow'][i, 0] == jacob[3]

            # psf stuff
            cen = pex.get_center(orig_row, orig_col)
            sigma = pex.get_sigma(orig_row, orig_col)
            assert np.allclose(obj['psf_cutout_row'][i, 0], cen[0])
            assert np.allclose(obj['psf_cutout_col'][i, 0], cen[1])
            assert np.allclose(obj['psf_sigma'][i, 0], sigma)
            assert obj['psf_start_row'][i, 0] == -9999
            assert obj['psf_box_size'][i] == -9999

        for tpe in ['image', 'weight', 'seg', 'bmask', 'noise']:
            if tpe == 'image':
                im = data[tpe] - data['bkg']
            else:
                im = data[tpe]
            for i in range(data['nobj']):
                cutout = m.get_cutout(i, 0, type=tpe)

                if tpe in ['seg', 'weight']:
                    assert np.allclose(
                        cutout,
                        im[
                            obj['orig_start_row'][i, 0]:
                            obj['orig_start_row'][i, 0] + 75,
                            obj['orig_start_col'][i, 0]:
                            obj['orig_start_col'][i, 0] + 75])
                elif tpe in ['bmask']:
                    assert np.allclose(cutout, bmask_gen.get_bmask(i))
                elif tpe in ['image']:
                    # check no interp
                    iimage = im[
                        obj['orig_start_row'][i, 0]:
                        obj['orig_start_row'][i, 0] + 75,
                        obj['orig_start_col'][i, 0]:
                        obj['orig_start_col'][i, 0] + 75]
                    ct = m.get_cutout_nomasking(i, 0, type='image')
                    assert np.allclose(iimage, ct)

                    # check the noise interp
                    bmsk = bmask_gen.get_bmask(i)
                    msk = (
                        bmsk &
                        data['config']['masking_config']['noise_interp_flags'])
                    msk = msk != 0
                    inoise = m.get_cutout_nomasking(
                        i, 0, type='noise_for_noise_interp')
                    assert np.allclose(cutout[msk], inoise[msk])

                    # check the se interp
                    bmsk = bmask_gen.get_bmask(i)
                    msk = (
                        bmsk &
                        data['config']['masking_config']['se_interp_flags'])
                    msk = msk != 0
                    assert np.allclose(cutout[msk], ct[msk])
                elif tpe in ['noise']:
                    # check no interp
                    iimage = im[
                        obj['orig_start_row'][i, 0]:
                        obj['orig_start_row'][i, 0] + 75,
                        obj['orig_start_col'][i, 0]:
                        obj['orig_start_col'][i, 0] + 75]
                    ct = m.get_cutout_nomasking(i, 0, type='noise')
                    assert np.allclose(iimage, ct)

                    bmsk = bmask_gen.get_bmask(i)

                    # check that the noise is different where an interpolation
                    # was applied
                    msk = (
                        bmsk &
                        data['config']['masking_config']['se_interp_flags'])
                    msk = msk != 0
                    inoise = m.get_cutout_nomasking(
                        i, 0, type='noise')
                    assert not np.allclose(cutout[msk], inoise[msk])
                else:
                    assert False

        # make sure the caching works
        # we should only have as many misses as objects since we called it
        # a loop
        assert m._cached_masking_func.cache_info().misses == data['nobj']

        for i in range(data['nobj']):
            cutout = m.get_cutout(i, 0, type='psf')

            assert np.allclose(
                cutout,
                pex.get_rec(obj['orig_row'][i, 0], obj['orig_col'][i, 0]))

        ii = m.get_image_info()
        for tag in ['image', 'weight', 'seg', 'bmask', 'bkg']:
            for tail in ['path', 'ext']:
                col = '%s_%s' % (tag, tail)
                if tail == 'path':
                    val = bytes(ii[col]).decode('utf-8').rstrip(' \t\r\n\0')
                else:
                    val = ii[col][0]
                assert val == data['config'][col]
        assert ii['image_id'] == 0
        assert ii['image_flags'] == 0
        assert ii['magzp'] == 30.0
        assert ii['scale'] == 10.0**(0.4*(MAGZP_REF - 30.0))
        assert ii['position_offset'] == POSITION_OFFSET
        ii_wcs = json.loads(ii['wcs'][0].decode('utf=8'))
        for k, v in data['wcs_header'].items():
            assert ii_wcs[k.lower()] == v

        metadata = m.get_meta()
        assert metadata['magzp_ref'] == MAGZP_REF
