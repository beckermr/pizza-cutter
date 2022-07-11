import copy
import yaml

import pytest

from .._des_info import check_info, flag_data_in_info


INFO_YAML = """\
band: z
bmask_ext: msk
bmask_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/coadd/DES2005-5123_r4575p01_z.fits.fz
cat_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/cat/DES2005-5123_r4575p01_z_cat.fits
compression: .fz
filename: DES2005-5123_r4575p01_z.fits
image_ext: sci
image_flags: 0
image_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/coadd/DES2005-5123_r4575p01_z.fits.fz
image_shape:
- 10000
- 10000
magzp: 30.0
path: OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/coadd
pfw_attempt_id: 2730721
position_offset: 1
psf_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/psf/DES2005-5123_r4575p01_z_psfcat.psf
scale: 1.0
seg_ext: sci
seg_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/seg/DES2005-5123_r4575p01_z_segmap.fits
gaia_stars_file: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/cal/cat_tile_gaia/v1/DES2005-5123_GAIA_DR2_v1.fits
src_info:
- band: z
  bkg_ext: sci
  bkg_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/red/bkg/D00473830_z_c05_r4433p01_bkg.fits.fz
  bmask_ext: msk
  bmask_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/red/immask/D00473830_z_c05_r4433p01_immasked.fits.fz
  ccdnum: 5
  compression: .fz
  expnum: 473830
  filename: D00473830_z_c05_r4433p01_immasked.fits
  head_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/aux/DES2005-5123_r4575p01_D00473830_z_c05_scamp.ohead
  image_ext: sci
  image_flags: 0
  image_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/red/immask/D00473830_z_c05_r4433p01_immasked.fits.fz
  image_shape:
  - 4096
  - 2048
  magzp: 31.292797088623047
  path: OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/red/immask
  pfw_attempt_id: 2730721
  piff_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1_PIFF/20150911-r5018/D00473830/p01/psf/D00473830_z_c05_r5018p01_piff-model.fits
  position_offset: 1
  psf_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/psf/D00473830_z_c05_r4433p01_psfexcat.psf
  psfex_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/psf/D00473830_z_c05_r4433p01_psfexcat.psf
  scale: 0.30400530659860264
  seg_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/seg/D00473830_z_c05_r4433p01_segmap.fits.fz
  tilename: DES2005-5123
  weight_ext: wgt
  weight_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1/r4433/20150911/D00473830/p01/red/immask/D00473830_z_c05_r4433p01_immasked.fits.fz
  piff_info:
    desdm_flags: 0
    fwhm_cen: 2.0
    star_t_std: 0.03
    star_t_mean: 0.5
    nstar: 55
    exp_star_t_mean: 0.55
    exp_star_t_std: 0.02
- band: z
  bkg_ext: sci
  bkg_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/red/bkg/D00675122_z_c56_r3515p01_bkg.fits.fz
  bmask_ext: msk
  bmask_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/red/immask/D00675122_z_c56_r3515p01_immasked.fits.fz
  ccdnum: 56
  compression: .fz
  expnum: 675122
  filename: D00675122_z_c56_r3515p01_immasked.fits
  head_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/multiepoch/Y6A1/r4575/DES2005-5123/p01/aux/DES2005-5123_r4575p01_D00675122_z_c56_scamp.ohead
  image_ext: sci
  image_flags: 0
  image_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/red/immask/D00675122_z_c56_r3515p01_immasked.fits.fz
  image_shape:
  - 4096
  - 2048
  magzp: 31.4688777923584
  path: OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/red/immask
  pfw_attempt_id: 2730721
  piff_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y6A1_PIFF/20170906-r5022/D00675122/p01/psf/D00675122_z_c56_r5022p01_piff-model.fits
  position_offset: 1
  psf_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/psf/D00675122_z_c56_r3515p01_psfexcat.psf
  psfex_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/psf/D00675122_z_c56_r3515p01_psfexcat.psf
  scale: 0.2584930572454005
  seg_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/seg/D00675122_z_c56_r3515p01_segmap.fits.fz
  tilename: DES2005-5123
  weight_ext: wgt
  weight_path: /Users/beckermr/MEDS_DIR/des-pizza-slices-y6-v6/DES2005-5123/sources-z/OPS/finalcut/Y5A1/r3515/20170906/D00675122/p01/red/immask/D00675122_z_c56_r3515p01_immasked.fits.fz
  piff_info:
    desdm_flags: 10
    fwhm_cen: 2.0
    star_t_std: 0.03
    star_t_mean: 0.5
    nstar: 55
    exp_star_t_mean: 0.55
    exp_star_t_std: 0.02
"""  # noqa


def test_flag_data_in_info():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))
    flag_data_in_info(
        info=info,
        config={
            "single_epoch": {
                "piff_cuts": dict(
                    max_fwhm_cen=3,
                    min_nstar=25,
                    max_exp_T_mean_fac=4,
                    max_ccd_T_std_fac=0.3,
                ),
            },
        },
    )
    assert info["src_info"][0]["image_flags"] == 0
    assert info["src_info"][1]["image_flags"] == 2**0


def test_check_info_smoke():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))
    check_info(info=info)


def test_check_info_coadd_paths():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))
    coadd_keys = [
        "image_path", "seg_path", "bmask_path",
        "gaia_stars_file", "psf_path",
    ]
    for key in coadd_keys:
        _info = copy.deepcopy(info)
        _info[key] = _info[key].replace("DES2005-5123", "DES2005-5823")
        with pytest.raises(RuntimeError) as e:
            check_info(info=_info)
        assert _info[key] in str(e.value)


def test_check_info_coadd_scale():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))
    info["scale"] = 2.0
    with pytest.raises(RuntimeError) as e:
        check_info(info=info)
    assert "coadd image scale" in str(e.value)


def test_check_info_band_entries():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))

    _info = copy.deepcopy(info)
    _info["band"] = "a"
    with pytest.raises(RuntimeError) as e:
        check_info(info=_info)
    assert "band entries do not all match" in str(e.value)

    _info = copy.deepcopy(info)
    _info["src_info"][0]["band"] = "a"
    with pytest.raises(RuntimeError) as e:
        check_info(info=_info)
    assert "band entries do not all match" in str(e.value)


def test_check_info_coadd_band():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))
    band = info["band"]
    ends = dict(
        bmask_path=f"_{band}.fits.fz",
        cat_path=f"_{band}_cat.fits",
        image_path=f"_{band}.fits.fz",
        psf_path=f"_{band}_psfcat.psf",
        seg_path=f"_{band}_segmap.fits",
    )

    for key, end in ends.items():
        _info = copy.deepcopy(info)
        _info[key] = _info[key].replace(end, end.replace(f"_{band}", "_a"))
        with pytest.raises(RuntimeError) as e:
            check_info(info=_info)
        assert f"doesn't end with ['{end}" in str(e.value)


def test_check_info_se_files():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))

    se_keys = [
        "bkg_path",
        "bmask_path",
        "image_path",
        "piff_path",
        "psfex_path",
        "psf_path",
        "seg_path",
        "weight_path",
    ]
    band = info["band"]
    for i in range(len(info["src_info"])):
        ii = info["src_info"][i]
        ccd_slug = "D%08d_%s_c%02d_" % (ii["expnum"], band, ii["ccdnum"])
        for key in se_keys:
            _info = copy.deepcopy(info)
            _info["src_info"][i][key] = _info["src_info"][i][key].replace(
                ccd_slug, ccd_slug.replace(f"_{band}", "_a")
            )
            with pytest.raises(RuntimeError) as e:
                check_info(info=_info)
            assert f"doesn't start with {ccd_slug}" in str(e.value)


def test_check_info_se_tilename():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))

    for i in range(len(info["src_info"])):
        _info = copy.deepcopy(info)
        _info["src_info"][i]["tilename"] = "blah"
        with pytest.raises(RuntimeError) as e:
            check_info(info=_info)
        assert "has the wrong tilename" in str(e.value)


def test_check_info_se_scamp_header():
    info = copy.deepcopy(yaml.safe_load(INFO_YAML))
    key = "head_path"

    for i in range(len(info["src_info"])):
        _info = copy.deepcopy(info)
        _info["src_info"][i][key] = _info["src_info"][i][key].replace(
            _info["src_info"][i]["tilename"], "blah"
        )
        with pytest.raises(RuntimeError) as e:
            check_info(info=_info)
        assert f"doesn't start with {_info['src_info'][i]['tilename']}" in str(e.value)

        _info = copy.deepcopy(info)
        scamp_slug = "_%s_c%02d_scamp.ohead" % (
            _info['src_info'][i]['band'],
            _info['src_info'][i]["ccdnum"],
        )
        _info["src_info"][i][key] = _info["src_info"][i][key].replace(
            scamp_slug, "blah"
        )
        with pytest.raises(RuntimeError) as e:
            check_info(info=_info)
        assert f"doesn't end with {scamp_slug}" in str(e.value)
