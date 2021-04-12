import os
import subprocess
import galsim
import yaml
import copy

import pytest
import numpy as np

import meds
from ..data import (
    generate_sim, write_sim,
    SIM_BMASK_SPLINE_INTERP,
    SIM_BMASK_NOISE_INTERP,
    SIM_CONFIG,
    SIM_CONFIG_ROTLIST,
)
from ..._constants import (
    MAGZP_REF, BMASK_SPLINE_INTERP, BMASK_NOISE_INTERP, BMASK_GAIA_STAR,
)
from ....slice_utils.procflags import (
    SLICE_HAS_FLAGS,
    HIGH_MASKED_FRAC)

from ..._affine_wcs import AffineWCS
from ..._coadd_slices import _get_gaia_stars
from ..._se_image import _compute_wcs_area


def _coadd_end2end(tmp_path_factory, sim_config):
    tmp_path = tmp_path_factory.getbasetemp()
    info, images, weights, bmasks, bkgs, _ = generate_sim()
    write_sim(
        path=tmp_path, info=info,
        images=images, weights=weights, bmasks=bmasks, bkgs=bkgs,
        gaia_stars=None,
    )

    with open(os.path.join(tmp_path, 'config.yaml'), 'w') as fp:
        fp.write(sim_config)

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
        cp = subprocess.run(cmd, check=True, shell=True, capture_output=True)
        stdout = str(cp.stdout, 'utf-8').replace('\\n', '\n')
        stderr = str(cp.stderr, 'utf-8').replace('\\n', '\n')
        print('stdout:\n', stdout)
        print('stderr:\n', stdout)
    except subprocess.CalledProcessError as err:
        stdout = str(err.stdout, 'utf-8').replace('\\n', '\n')
        stderr = str(err.stderr, 'utf-8').replace('\\n', '\n')
        print('stdout:\n', stdout)
        print('stderr:\n', stderr)
        raise
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
        'config': copy.deepcopy(sim_config),
    }


@pytest.fixture(scope="session")
def coadd_end2end(tmp_path_factory):
    return _coadd_end2end(tmp_path_factory, SIM_CONFIG)


@pytest.fixture(scope="session")
def coadd_end2end_rotlist(tmp_path_factory):
    return _coadd_end2end(tmp_path_factory, SIM_CONFIG_ROTLIST)


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
                ).shear(g1=0.1, g2=-0.1)
            )
    max_wgts = np.array(max_wgts)
    max_wgts = max_wgts / np.sum(max_wgts)

    psf_m = m.get_cutout(0, 0, type='psf')

    psfs = [psf.withFlux(wgt) for psf, wgt in zip(psfs, max_wgts)]
    psf_im = galsim.Sum(psfs).withFlux(np.sum(psf_m)).drawImage(
        nx=51, ny=51, scale=0.25).array

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(np.arcsinh(psf_im))
        axs[1].imshow(np.arcsinh(psf_m))
        import pdb
        pdb.set_trace()

    # we don't have a strict criterion here since the coadding process
    # broadens the PSF a bit. Instead we are setting tolerances and if code
    # changes make these break, then we need to look again.
    assert np.max(np.abs(psf_im - psf_m)) < 6e-4

    # we also demand that the FWHM is about the same
    print("fwhm true|coadd: %s|%s" % (
        galsim.ImageD(psf_im, scale=0.25).calculateFWHM(),
        galsim.ImageD(psf_m, scale=0.25).calculateFWHM())
    )
    assert np.abs(
        galsim.ImageD(psf_im, scale=0.25).calculateFWHM() -
        galsim.ImageD(psf_m, scale=0.25).calculateFWHM()) < 4e-3


def test_coadding_end2end_gal(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    weights = coadd_end2end['weights']
    info = coadd_end2end['info']
    ei = m._fits['epochs_info'].read()
    gal = galsim.Gaussian(fwhm=2.5).shear(g1=-0.2, g2=0.5)

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
                ).shear(g1=0.1, g2=-0.1)
            )
    max_wgts = np.array(max_wgts)
    max_wgts = max_wgts / np.sum(max_wgts)

    gal_m = m.get_cutout(0, 0, type='image')

    gal_psf = [
        galsim.Convolve(gal, psf.withFlux(wgt))
        for psf, wgt in zip(psfs, max_wgts)]
    gal_im = galsim.Sum(gal_psf).withFlux(np.sum(gal_m)).drawImage(
        nx=49, ny=49, scale=0.25).array

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(np.arcsinh(gal_im))
        axs[1].imshow(np.arcsinh(gal_m))
        import pdb
        pdb.set_trace()

    # we don't have a strict criterion here since the coadding process
    # broadens the PSF a bit. Instead we are setting tolerances and if code
    # changes make these break, then we need to look again.
    print("max abs diff:", np.max(np.abs(gal_im - gal_m)))
    assert np.max(np.abs(gal_im - gal_m)) < 5e-4

    # we also demand that the FWHM is about the same
    print("fwhm true|coadd: %s|%s" % (
        galsim.ImageD(gal_im, scale=0.25).calculateFWHM(),
        galsim.ImageD(gal_m, scale=0.25).calculateFWHM())
    )
    assert np.abs(
        galsim.ImageD(gal_im, scale=0.25).calculateFWHM() -
        galsim.ImageD(gal_m, scale=0.25).calculateFWHM()) < 1e-2

    # and the center
    y, x = np.mgrid[0:49, 0:49]
    print("xcen true|coadd: %s|%s" % (
        np.sum(x*gal_im)/np.sum(gal_im),
        np.sum(x*gal_m)/np.sum(gal_m),
    ))
    print("ycen true|coadd: %s|%s" % (
        np.sum(y*gal_im)/np.sum(gal_im),
        np.sum(y*gal_m)/np.sum(gal_m),
    ))
    assert np.abs(
        np.sum(x*gal_m)/np.sum(gal_m) -
        np.sum(x*gal_im)/np.sum(gal_im)) < 1.5e-1
    assert np.abs(
        np.sum(y*gal_m)/np.sum(gal_m) -
        np.sum(y*gal_im)/np.sum(gal_im)) < 1.5e-1

    # and shear
    mom_im = galsim.hsm.FindAdaptiveMom(galsim.ImageD(gal_im, scale=0.25))
    mom_m = galsim.hsm.FindAdaptiveMom(galsim.ImageD(gal_m, scale=0.25))
    print("e1 true|coadd: %s|%s" % (
        mom_im.observed_shape.e1,
        mom_m.observed_shape.e1,
    ))
    print("e2 true|coadd: %s|%s" % (
        mom_im.observed_shape.e2,
        mom_m.observed_shape.e2,
    ))
    assert np.abs(mom_im.observed_shape.e1 - mom_m.observed_shape.e1) < 0.005
    assert np.abs(mom_im.observed_shape.e2 - mom_m.observed_shape.e2) < 0.005


def _plot_it(bmask, flag=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=1, ncols=1)
    if flag is not None:
        axs.imshow((bmask & flag) != 0)
    else:
        axs.imshow(bmask)
    import pdb
    pdb.set_trace()


def test_coadding_end2end_masks(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    bmask = m.get_cutout(0, 0, type='bmask')
    ormask = m.get_cutout(0, 0, type='ormask')

    # somwhere in the middle spline interpolation was done
    if False:
        _plot_it(bmask)
    assert np.mean((bmask[:, 24:26] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((bmask[24:26, :] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[:, 24:26] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[24:26, :] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0

    assert np.mean((bmask[:, 18:20] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((bmask[18:20, :] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[:, 18:20] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[18:20, :] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0

    # we did some noise interp too
    assert np.mean((bmask[:, 33:35] & BMASK_NOISE_INTERP) != 0) > 0.0
    assert np.mean((bmask[33:35, :] & BMASK_NOISE_INTERP) != 0) > 0.0
    assert np.mean((ormask[:, 33:35] & SIM_BMASK_NOISE_INTERP) != 0) > 0.0
    assert np.mean((ormask[33:35, :] & SIM_BMASK_NOISE_INTERP) != 0) > 0.0


def test_coadding_end2end_masks_rotlist(coadd_end2end_rotlist):
    m = meds.MEDS(coadd_end2end_rotlist['meds_path'])
    bmask = m.get_cutout(0, 0, type='bmask')
    ormask = m.get_cutout(0, 0, type='ormask')

    # somwhere in the middle spline interpolation was done
    if False:
        _plot_it(bmask)  # , BMASK_SPLINE_INTERP)
    assert np.mean((bmask[:, 24:26] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((bmask[24:26, :] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[:, 24:26] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[24:26, :] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0

    assert np.mean((bmask[:, 18:20] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((bmask[18:20, :] & BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[:, 18:20] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0
    assert np.mean((ormask[18:20, :] & SIM_BMASK_SPLINE_INTERP) != 0) > 0.0

    # we did some noise interp too
    assert np.mean((bmask[:, 33:35] & BMASK_NOISE_INTERP) != 0) > 0.0
    assert np.mean((bmask[33:35, :] & BMASK_NOISE_INTERP) != 0) > 0.0
    assert np.mean((ormask[:, 33:35] & SIM_BMASK_NOISE_INTERP) != 0) > 0.0
    assert np.mean((ormask[33:35, :] & SIM_BMASK_NOISE_INTERP) != 0) > 0.0


def test_coadding_end2end_mfrac(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    mfrac = m.get_cutout(0, 0, type='mfrac')

    def _plot_it(iimg, **kwargs):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.imshow(iimg, **kwargs)
        import pdb
        pdb.set_trace()

    # somwhere in the middle interpolation was done
    if False:
        _plot_it(mfrac)

    assert np.any(mfrac[:, 24:26] > 0.0)
    assert np.any(mfrac[:, 18:20] > 0.0)
    assert np.any(mfrac[:, 33:35] > 0.0)
    assert np.all(mfrac[0:10, 0:10] == 0.0)
    # fpacking doesn't preserve the range [0, 1]
    assert np.all(mfrac >= -1e-3)
    assert np.all(mfrac <= 1.001)


def test_coadding_end2end_noise(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    weights = coadd_end2end['weights']
    info = coadd_end2end['info']
    ei = m._fits['epochs_info'].read()

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
    var = 1.0 / np.sum(max_wgts)
    nse = m.get_cutout(0, 0, type='noise')

    # due to interpolation causing correlations noise
    # variance should be smaller
    # we also demand that it matches to better than 20%
    assert np.std(nse) <= np.sqrt(var)
    assert np.allclose(np.std(nse), np.sqrt(var), atol=0, rtol=0.2)


def test_coadding_end2end_weight(coadd_end2end):
    m = meds.MEDS(coadd_end2end['meds_path'])
    wgt = m.get_cutout(0, 0, type='weight')
    nse = m.get_cutout(0, 0, type='noise')
    np.allclose(wgt, 1.0 / np.var(nse))


def test_coadding_end2end_gaia_stars(tmp_path_factory):
    """
    make sure pixels are getting masked
    """
    tmp_path = tmp_path_factory.getbasetemp()
    info, images, weights, bmasks, bkgs, gaia_stars = generate_sim()
    write_sim(
        path=tmp_path, info=info,
        images=images, weights=weights, bmasks=bmasks, bkgs=bkgs,
        gaia_stars=gaia_stars,
    )

    with open(os.path.join(tmp_path, 'config.yaml'), 'w') as fp:
        fp.write(SIM_CONFIG)

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
        cp = subprocess.run(cmd, check=True, shell=True, capture_output=True)
        stdout = str(cp.stdout, 'utf-8').replace('\\n', '\n')
        stderr = str(cp.stderr, 'utf-8').replace('\\n', '\n')
        print('stdout:\n', stdout)
        print('stderr:\n', stdout)
    except subprocess.CalledProcessError as err:
        stdout = str(err.stdout, 'utf-8').replace('\\n', '\n')
        stderr = str(err.stderr, 'utf-8').replace('\\n', '\n')
        print('stdout:\n', stdout)
        print('stderr:\n', stderr)
        raise
    finally:
        if mdir is not None:
            os.environ['MEDS_DIR'] = mdir
        else:
            del os.environ['MEDS_DIR']

    meds_path = os.path.join(
        tmp_path,
        'e2e_test_p_config_meds-pizza-slices.fits.fz',
    )

    m = meds.MEDS(meds_path)
    bmask = m.get_cutout(0, 0, type='bmask')
    mfrac = m.get_cutout(0, 0, type='mfrac')
    weight = m.get_cutout(0, 0, type='weight')

    # make sure some are masked
    assert np.any((bmask & BMASK_GAIA_STAR) != 0)

    wcs = AffineWCS(**info['affine_wcs_config'])

    config = yaml.load(SIM_CONFIG, Loader=yaml.SafeLoader)

    gaia_mask_config = config['gaia_star_masks']
    gaia_stars = _get_gaia_stars(
        fname=info['gaia_stars_file'], wcs=wcs,
        poly_coeffs=tuple(gaia_mask_config['poly_coeffs']),
        radius_factor=gaia_mask_config['radius_factor'],
        max_g_mag=gaia_mask_config['max_g_mag'],
        wcs_position_offset=1,
    )

    scale = np.sqrt(_compute_wcs_area(wcs, 10, 10))

    # check star center is masked
    # the 1.0 are not perfectly preserved in the compression. Only
    # zeros are perfectly preserved
    mfrac_tol = 0.001
    for star in gaia_stars:
        x, y = wcs.sky2image(star['ra'], star['dec'])
        ix = int(x)
        iy = int(y)

        assert bmask[iy, ix] & BMASK_GAIA_STAR != 0
        assert weight[iy, ix] == 0.0
        assert abs(mfrac[iy, ix] - 1.0) < mfrac_tol

        ixlow = int(x - star['radius_arcsec']/scale + 2)
        if ixlow < 0:
            ixlow = 0
        assert bmask[iy, ixlow] & BMASK_GAIA_STAR != 0
        assert weight[iy, ixlow] == 0.0
        assert abs(mfrac[iy, ixlow] - 1.0) < mfrac_tol

        ixhigh = int(x + star['radius_arcsec']/scale - 3)
        if ixhigh > bmask.shape[1] - 1:
            ixhigh = bmask.shape[1] - 1
        assert bmask[iy, ixhigh] & BMASK_GAIA_STAR != 0
        assert weight[iy, ixhigh] == 0.0
        assert abs(mfrac[iy, ixhigh] - 1.0) < mfrac_tol

        iylow = int(y - star['radius_arcsec']/scale + 2)
        if iylow < 0:
            iylow = 0
        assert bmask[iylow, ix] & BMASK_GAIA_STAR != 0
        assert weight[iylow, ix] == 0.0
        assert abs(mfrac[iylow, ix] - 1.0) < mfrac_tol

        iyhigh = int(y + star['radius_arcsec']/scale - 3)
        if iyhigh > bmask.shape[1] - 1:
            iyhigh = bmask.shape[1] - 1
        assert bmask[iyhigh, ix] & BMASK_GAIA_STAR != 0
        assert weight[iyhigh, ix] == 0.0
        assert abs(mfrac[iyhigh, ix] - 1.0) < mfrac_tol


def test_coadding_end2end_range_kwarg(tmp_path_factory):
    tmp_path = tmp_path_factory.getbasetemp()
    info, images, weights, bmasks, bkgs, gaia_stars = generate_sim()
    write_sim(
        path=tmp_path, info=info,
        images=images, weights=weights, bmasks=bmasks, bkgs=bkgs,
        gaia_stars=gaia_stars,
    )

    with open(os.path.join(tmp_path, 'config.yaml'), 'w') as fp:
        fp.write(SIM_CONFIG)

    cmd = """\
    des-pizza-cutter \
      --config={config} \
      --info={info} \
      --output-path={tmp_path} \
      --log-level=DEBUG \
      --range 0:1 \
      --seed=42
    """.format(
        config=os.path.join(tmp_path, 'config.yaml'),
        info=os.path.join(tmp_path, 'info.yaml'),
        tmp_path=tmp_path)

    mdir = os.environ.get('MEDS_DIR')
    try:
        os.environ['MEDS_DIR'] = 'meds_dir_xyz'
        cp = subprocess.run(cmd, check=True, shell=True, capture_output=True)
        stdout = str(cp.stdout, 'utf-8').replace('\\n', '\n')
        stderr = str(cp.stderr, 'utf-8').replace('\\n', '\n')
        print('stdout:\n', stdout)
        print('stderr:\n', stdout)
    except subprocess.CalledProcessError as err:
        stdout = str(err.stdout, 'utf-8').replace('\\n', '\n')
        stderr = str(err.stderr, 'utf-8').replace('\\n', '\n')
        print('stdout:\n', stdout)
        print('stderr:\n', stderr)
        raise
    finally:
        if mdir is not None:
            os.environ['MEDS_DIR'] = mdir
        else:
            del os.environ['MEDS_DIR']

    meds_path = os.path.join(
        tmp_path,
        'e2e_test_p_config_meds-pizza-slices.fits.fz',
    )

    m = meds.MEDS(meds_path)
    object_data = m._fits['object_data'].read()
    assert len(object_data) == 1
