import os
import subprocess
import galsim
import copy
import yaml

import pytest
import numpy as np

import meds
from ..data import (
    generate_sim, write_sim,
    SIM_CONFIG,
)


GAL_FWHM = 1e-10


def _coadd_end2end_star(tmp_path_factory, sim_config):
    tmp_path = tmp_path_factory.getbasetemp()
    info, images, weights, bmasks, bkgs, gaia_stars = generate_sim(
        gal_fwhm=GAL_FWHM, artifacts=False,
    )
    write_sim(
        path=tmp_path, info=info,
        images=images, weights=weights, bmasks=bmasks, bkgs=bkgs,
        gaia_stars=gaia_stars,
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
        tmp_path=tmp_path
    )

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
            'e2e-test_a_config_meds-pizza-slices.fits.fz'),
        'images': images,
        'weights': weights,
        'bmasks': bmasks,
        'info': info,
        'bkgs': bkgs,
        'config': copy.deepcopy(sim_config),
        'config_yaml': yaml.safe_load(copy.deepcopy(sim_config)),
    }


@pytest.fixture(scope="session")
def coadd_end2end_star(tmp_path_factory):
    return _coadd_end2end_star(tmp_path_factory, SIM_CONFIG)


def test_coadding_end2end_gal_is_star(coadd_end2end_star):
    m = meds.MEDS(coadd_end2end_star['meds_path'])

    psf_m = m.get_cutout(0, 0, type='psf')
    gal_m = m.get_cutout(0, 0, type='image')

    cut = abs(psf_m.shape[0] - gal_m.shape[0]) // 2
    if psf_m.shape[0] < gal_m.shape[0]:
        gal_m = gal_m[cut:-cut, cut:-cut]
    elif psf_m.shape[0] > gal_m.shape[0]:
        psf_m = psf_m[cut:-cut, cut:-cut]

    gal_m = gal_m / np.sum(gal_m)
    psf_m = psf_m / np.sum(psf_m)

    dim = psf_m.shape[0]
    assert psf_m.shape == gal_m.shape

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=1, ncols=3)
        axs[0].imshow(np.arcsinh(gal_m))
        axs[1].imshow(np.arcsinh(psf_m))
        axs[2].imshow(np.arcsinh(gal_m - psf_m))
        import pdb
        pdb.set_trace()

    print("max abs diff:", np.max(np.abs(gal_m - psf_m)))
    assert np.max(np.abs(gal_m - psf_m)) < 1e-6

    # we also demand that the FWHM is about the same
    print("fwhm gal|psf: %s|%s" % (
        galsim.ImageD(gal_m, scale=0.25).calculateFWHM(),
        galsim.ImageD(psf_m, scale=0.25).calculateFWHM())
    )
    assert np.abs(
        galsim.ImageD(gal_m, scale=0.25).calculateFWHM() -
        galsim.ImageD(psf_m, scale=0.25).calculateFWHM()) < 1e-5

    # and the center
    y, x = np.mgrid[0:dim, 0:dim]
    print("xcen gal|psf: %s|%s" % (
        np.sum(x*gal_m)/np.sum(gal_m),
        np.sum(x*psf_m)/np.sum(psf_m),
    ))
    print("ycen gal|psf: %s|%s" % (
        np.sum(y*gal_m)/np.sum(gal_m),
        np.sum(y*psf_m)/np.sum(psf_m),
    ))
    assert np.abs(
        np.sum(x*gal_m)/np.sum(gal_m) -
        np.sum(x*psf_m)/np.sum(psf_m)) < 1e-4
    assert np.abs(
        np.sum(y*gal_m)/np.sum(gal_m) -
        np.sum(y*psf_m)/np.sum(psf_m)) < 1e-4

    # and shear
    mom_gal = galsim.hsm.FindAdaptiveMom(galsim.ImageD(gal_m, scale=0.25))
    mom_psf = galsim.hsm.FindAdaptiveMom(galsim.ImageD(psf_m, scale=0.25))
    print("e1 gal|psf: %s|%s" % (
        mom_gal.observed_shape.e1,
        mom_psf.observed_shape.e1,
    ))
    print("e2 gal|psf: %s|%s" % (
        mom_gal.observed_shape.e2,
        mom_psf.observed_shape.e2,
    ))
    assert np.abs(mom_gal.observed_shape.e1 - mom_psf.observed_shape.e1) < 1e-5
    assert np.abs(mom_gal.observed_shape.e2 - mom_psf.observed_shape.e2) < 1e-5
