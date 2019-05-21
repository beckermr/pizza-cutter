import os
import subprocess
import galsim

import pytest
import numpy as np

import meds
from ..data import (
    generate_sim, write_sim,
    SIM_BMASK_SPLINE_INTERP,
    SIM_BMASK_NOISE_INTERP)
from ..._constants import MAGZP_REF, BMASK_SPLINE_INTERP, BMASK_NOISE_INTERP
from ....slice_utils.procflags import (
    SLICE_HAS_FLAGS,
    HIGH_MASKED_FRAC)


@pytest.fixture(scope="session")
def coadd_end2end(tmp_path_factory):
    tmp_path = tmp_path_factory.getbasetemp()
    info, images, weights, bmasks, bkgs = generate_sim()
    write_sim(path=tmp_path, info=info,
              images=images, weights=weights, bmasks=bmasks, bkgs=bkgs)

    config = """\
# optional but these are good defaults
fpack_pars:
  FZQVALUE: 4
  FZTILE: "(10240,1)"
  FZALGOR: "RICE_1"
  # preserve zeros, don't dither them
  FZQMETHD: "SUBTRACTIVE_DITHER_2"

coadd:
  # these are in pixels
  # the total "pizza slice" will be central_size + 2 * buffer_size
  central_size: 33  # size of the central region
  buffer_size: 8  # size of the buffer on each size

  psf_box_size: 51

  wcs_type: affine
  coadding_weight: 'noise'

single_epoch:
  psf_type: galsim
  wcs_type: affine

  reject_outliers: False
  symmetrize_masking: True
  max_masked_fraction: 0.1
  max_unmasked_trail_fraction: 0.02
  edge_buffer: 8

  mask_tape_bumps: False

  spline_interp_flags:
    - 2

  noise_interp_flags:
    - 4

  bad_image_flags:
    - 1
"""

    with open(os.path.join(tmp_path, 'config.yaml'), 'w') as fp:
        fp.write(config)

    cmd = """\
    des-pizza-cutter \
      --config={config} \
      --info={info} \
      --output-path={tmp_path} \
      --log-level=DEBUG \
      --seed=42
    """.format(
        config=os.path.join(tmp_path, 'config.yaml'),
        info=os.path.join(tmp_path, 'info.yaml'),
        tmp_path=tmp_path)

    mdir = os.environ.get('MEDS_DIR')
    try:
        os.environ['MEDS_DIR'] = 'meds_dir_xyz'
        subprocess.run(cmd, check=True, shell=True)
    finally:
        if mdir is not None:
            os.environ['MEDS_DIR'] = mdir
        else:
            del os.environ['MEDS_DIR']

    return {
        'meds_path': os.path.join(
            tmp_path,
            'e2e_test_p_config_meds-pizza-slices.fits.fz'),
        'images': images,
        'weights': weights,
        'bmasks': bmasks,
        'info': info,
        'bkgs': bkgs,
        'config': config,
    }


def test_coadding_end2end_epochs_info(coadd_end2end):
    weights = coadd_end2end['weights']
    info = coadd_end2end['info']
    m = meds.MEDS(coadd_end2end['meds_path'])

    ei = m._fits['epochs_info'].read()

    ############################################################
    # check the epochs info for correct flags
    # images 0, 1, 2 do not intersect the slice or get cut before they make
    # it into the epochs_info extensions
    # we add 1 for the file id's
    assert np.all(ei['file_id'] != 1)
    assert np.all(ei['file_id'] != 2)
    assert np.all(ei['file_id'] != 3)

    # images 4 is flagged at the pixel level
    msk = ei['file_id'] == 4
    assert np.all((ei['flags'][msk] & SLICE_HAS_FLAGS) != 0)
    assert np.all(ei['weight'][msk] == 0)

    # images  6, 7, 8 have too high a masked fraction
    msk = ei['file_id'] == 6
    assert np.all((ei['flags'][msk] & HIGH_MASKED_FRAC) != 0)
    assert np.all(ei['weight'][msk] == 0)

    msk = ei['file_id'] == 7
    assert np.all((ei['flags'][msk] & HIGH_MASKED_FRAC) != 0)
    assert np.all(ei['weight'][msk] == 0)

    msk = ei['file_id'] == 8
    assert np.all((ei['flags'][msk] & HIGH_MASKED_FRAC) != 0)
    assert np.all(ei['weight'][msk] == 0)

    ############################################################
    # check the epochs info for correct weights
    max_wgts = []
    for ind in range(len(ei)):
        if ei['weight'][ind] > 0:
            src_ind = ei['file_id'][ind]-1
            max_wgts.append(
                np.max(weights[src_ind]) /
                info['src_info'][src_ind]['scale'] ** 2
                )
    max_wgts = np.array(max_wgts)
    max_wgts /= np.sum(max_wgts)
    loc = 0
    for ind in range(len(ei)):
        if ei['weight'][ind] > 0:
            assert np.allclose(max_wgts[loc], ei['weight'][ind]), (
                ind, max_wgts[loc], ei['weight'][ind])
            loc += 1


def test_coadding_end2end_metadata(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    metadata = m._fits['metadata'].read()

    # we only check a few things here...
    assert metadata['config'][0] == coadd_end2end['config']
    assert metadata['numpy_version'][0] == np.__version__
    assert metadata['meds_dir'][0] == 'meds_dir_xyz'


def test_coadding_end2end_image_info(coadd_end2end):
    info = coadd_end2end['info']
    m = meds.MEDS(coadd_end2end['meds_path'])
    image_info = m._fits['image_info'].read()

    assert image_info['magzp'][0] == MAGZP_REF
    assert image_info['scale'][0] == 1.0
    assert image_info['position_offset'][0] == 0

    for ind in range(len(info['src_info'])):
        for key in ['scale', 'magzp', 'position_offset']:
            assert np.allclose(
                image_info[key][ind + 1], info['src_info'][ind][key])
        for key in ['image_path', 'image_ext', 'weight_path', 'weight_ext',
                    'bmask_path', 'bmask_ext', 'bkg_path', 'bkg_ext']:
            assert image_info[key][ind + 1] == info['src_info'][ind][key]

    assert np.all(
        image_info['image_flags'] ==
        np.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))


def test_coadding_end2end_object_data(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    weights = coadd_end2end['weights']
    info = coadd_end2end['info']
    ei = m._fits['epochs_info'].read()
    object_data = m._fits['object_data'].read()

    ############################################################
    # get weights for computing nepoch_eff
    max_wgts = []
    for ind in range(len(ei)):
        if ei['weight'][ind] > 0:
            src_ind = ei['file_id'][ind]-1
            max_wgts.append(
                np.max(weights[src_ind]) /
                info['src_info'][src_ind]['scale'] ** 2
                )
    max_wgts = np.array(max_wgts)
    nepoch_eff = max_wgts.sum() / max_wgts.max()

    assert len(object_data) == 1
    assert object_data['box_size'][0] == 49
    assert object_data['ra'][0] == 0
    assert object_data['dec'][0] == 0
    assert object_data['ncutout'][0] == 1
    assert object_data['file_id'][0, 0] == 0
    assert object_data['cutout_row'][0, 0] == 24
    assert object_data['cutout_col'][0, 0] == 24

    assert object_data['orig_row'][0, 0] == 24
    assert object_data['orig_col'][0, 0] == 24
    assert object_data['orig_start_row'][0, 0] == 0
    assert object_data['orig_start_col'][0, 0] == 0

    assert object_data['dudrow'][0, 0] == 0
    assert object_data['dudcol'][0, 0] == 0.25

    assert object_data['dvdrow'][0, 0] == 0.25
    assert object_data['dvdcol'][0, 0] == 0

    assert object_data['nepoch'][0] == 8
    assert np.allclose(object_data['nepoch_eff'][0], nepoch_eff)
    assert object_data['psf_box_size'][0] == 51
    assert object_data['psf_cutout_col'][0, 0] == 25
    assert object_data['psf_cutout_row'][0, 0] == 25


def test_coadding_end2end_psf(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    weights = coadd_end2end['weights']
    info = coadd_end2end['info']
    ei = m._fits['epochs_info'].read()

    ############################################################
    # get weights for computing nepoch_eff
    max_wgts = []
    psfs = []
    for ind in range(len(ei)):
        if ei['weight'][ind] > 0:
            src_ind = ei['file_id'][ind]-1
            max_wgts.append(
                np.max(weights[src_ind]) /
                info['src_info'][src_ind]['scale'] ** 2
                )
            psfs.append(
                galsim.Gaussian(
                    fwhm=info['src_info'][src_ind]['galsim_psf_config']['fwhm']
                )
            )
    max_wgts = np.array(max_wgts)
    max_wgts = max_wgts / np.sum(max_wgts)

    psf_m = m.get_cutout(0, 0, type='psf')

    psfs = [psf.withFlux(wgt) for psf, wgt in zip(psfs, max_wgts)]
    psf_im = galsim.Sum(psfs).withFlux(np.sum(psf_m)).drawImage(
        nx=51, ny=51, scale=0.25).array

    # we don't have a strict criterion here since the coadding process
    # broadens the PSF a bit. Instead we are setting tolerances and if code
    # changes make these break, then we need to look again.
    assert np.max(np.abs(psf_im - psf_m))/np.max(psf_im) < 0.003

    # we also demand that the FWHM is about the same
    assert np.abs(
        galsim.ImageD(psf_im, scale=0.25).calculateFWHM() -
        galsim.ImageD(psf_m, scale=0.25).calculateFWHM()) < 1e-4


def test_coadding_end2end_gal(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    weights = coadd_end2end['weights']
    info = coadd_end2end['info']
    ei = m._fits['epochs_info'].read()
    gal = galsim.Gaussian(fwhm=0.5).shear(g1=0, g2=0.5)

    ############################################################
    # get weights for computing nepoch_eff
    max_wgts = []
    psfs = []
    for ind in range(len(ei)):
        if ei['weight'][ind] > 0:
            src_ind = ei['file_id'][ind]-1
            max_wgts.append(
                np.max(weights[src_ind]) /
                info['src_info'][src_ind]['scale'] ** 2
                )
            psfs.append(
                galsim.Gaussian(
                    fwhm=info['src_info'][src_ind]['galsim_psf_config']['fwhm']
                )
            )
    max_wgts = np.array(max_wgts)
    max_wgts = max_wgts / np.sum(max_wgts)

    gal_m = m.get_cutout(0, 0, type='image')

    gal_psf = [
        galsim.Convolve(gal, psf.withFlux(wgt))
        for psf, wgt in zip(psfs, max_wgts)]
    gal_im = galsim.Sum(gal_psf).withFlux(np.sum(gal_m)).drawImage(
        nx=49, ny=49, scale=0.25).array

    # we don't have a strict criterion here since the coadding process
    # broadens the PSF a bit. Instead we are setting tolerances and if code
    # changes make these break, then we need to look again.
    assert np.max(np.abs(gal_im - gal_m))/np.max(gal_im) < 0.03

    # we also demand that the FWHM is about the same
    assert np.abs(
        galsim.ImageD(gal_im, scale=0.25).calculateFWHM() -
        galsim.ImageD(gal_m, scale=0.25).calculateFWHM()) < 1e-2

    # and the center
    y, x = np.mgrid[0:49, 0:49]
    assert np.abs(
        np.sum(x*gal_m)/np.sum(gal_m) -
        np.sum(x*gal_im)/np.sum(gal_im)) < 1e-1
    assert np.abs(
        np.sum(y*gal_m)/np.sum(gal_m) -
        np.sum(y*gal_im)/np.sum(gal_im)) < 1e-1

    # and shear
    mom_im = galsim.hsm.FindAdaptiveMom(galsim.ImageD(gal_im, scale=0.25))
    mom_m = galsim.hsm.FindAdaptiveMom(galsim.ImageD(gal_m, scale=0.25))
    assert np.abs(mom_im.observed_shape.e1 - mom_m.observed_shape.e1) < 0.005
    assert np.abs(mom_im.observed_shape.e2 - mom_m.observed_shape.e2) < 0.005


def test_coadding_end2end_masks(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    bmask = m.get_cutout(0, 0, type='bmask')
    ormask = m.get_cutout(0, 0, type='ormask')

    # somwhere in the middle spline interpolation was done
    assert np.mean((bmask[:, 24:26] & BMASK_SPLINE_INTERP) != 0) > 0.8
    assert np.mean((bmask[24:26, :] & BMASK_SPLINE_INTERP) != 0) > 0.8
    assert np.mean((ormask[:, 24:26] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.8
    assert np.mean((ormask[24:26, :] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.8

    assert np.mean((bmask[:, 18:20] & BMASK_SPLINE_INTERP) != 0) > 0.8
    assert np.mean((bmask[18:20, :] & BMASK_SPLINE_INTERP) != 0) > 0.8
    assert np.mean((ormask[:, 18:20] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.8
    assert np.mean((ormask[18:20, :] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.8

    # we did some noise interp too
    assert np.mean((bmask[:, 33:35] & BMASK_NOISE_INTERP) != 0) > 0.8
    assert np.mean((bmask[33:35, :] & BMASK_NOISE_INTERP) != 0) > 0.8
    assert np.mean((ormask[:, 33:35] & SIM_BMASK_NOISE_INTERP) != 0) > 0.8
    assert np.mean((ormask[33:35, :] & SIM_BMASK_NOISE_INTERP) != 0) > 0.8
