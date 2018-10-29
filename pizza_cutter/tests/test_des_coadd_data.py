import os
import pytest

from ..des_coadd_data import DESCoadd, DESCoaddSources


@pytest.mark.skipif(
    os.environ.get('CI', 'false') == 'true',
    reason='DES coadd classes can only be tested with database access.')
def test_descoadd(tmpdir):
    try:
        band = 'r'
        tilename = 'DES0417-5914'
        os.environ['MEDS_DIR'] = str(tmpdir)
        medsconf = 'test_des_coadd_data'
        coadd = DESCoadd(
            medsconf=medsconf, tilename=tilename, band=band,
            campaign='Y3A1_COADD')
        info = coadd.get_info()

        assert info['tilename'] == 'DES0417-5914'
        assert info['filename'] == 'DES0417-5914_r2683p01_r.fits'
        assert info['compression'] == '.fz'
        assert (
            info['path'] == 'OPS/multiepoch/Y3A1/r2683/DES0417-5914/p01/coadd')
        assert info['band'] == 'r'
        assert info['pfw_attempt_id'] == 613593
        assert info['magzp'] == 30.0
        assert info['image_path'].endswith(
            'test_des_coadd_data/DES0417-5914/sources-r/OPS/multiepoch/Y3A1'
            '/r2683/DES0417-5914/p01/coadd/DES0417-5914_r2683p01_r.fits.fz')
        assert info['cat_path'].endswith(
            'test_descoadd0/test_des_coadd_data/DES0417-5914/sources-r/OPS'
            '/multiepoch/Y3A1/r2683/DES0417-5914/p01/cat/'
            'DES0417-5914_r2683p01_r_cat.fits')
        assert info['seg_path'].endswith(
            'test_descoadd0/test_des_coadd_data/DES0417-5914/sources-r'
            '/OPS/multiepoch/Y3A1/r2683/DES0417-5914/p01/seg/'
            'DES0417-5914_r2683p01_r_segmap.fits')
        assert info['psf_path'].endswith(
            'test_des_coadd_data/DES0417-5914/sources-r/OPS/multiepoch/Y3A1'
            '/r2683/DES0417-5914/p01/psf/DES0417-5914_r2683p01_r_psfcat.psf')
    except Exception:
        os.environ.pop('MEDS_DIR', None)
        raise


@pytest.mark.skipif(
    os.environ.get('CI', 'false') == 'true',
    reason='DES coadd classes can only be tested with database access.')
def test_descoaddsources(tmpdir):
    try:
        band = 'r'
        tilename = 'DES0417-5914'
        os.environ['MEDS_DIR'] = str(tmpdir)
        medsconf = 'test_des_coadd_data'
        coaddsrcs = DESCoaddSources(
            medsconf=medsconf, tilename=tilename, band=band,
            campaign='Y3A1_COADD')
        info = coaddsrcs.get_info()

        assert coaddsrcs.finalcut_campaign == 'Y3A1_FINALCUT'
        assert len(info) == 65
        assert all(i['tilename'] == tilename for i in info)
        assert all(i['band'] == band for i in info)
        assert all(i['compression'] == '.fz' for i in info)
    except Exception:
        os.environ.pop('MEDS_DIR', None)
        raise


@pytest.mark.skipif(
    os.environ.get('CI', 'false') == 'true',
    reason='DES coadd classes can only be tested with database access.')
def test_descoadd_and_descoaddsources(tmpdir):
    try:
        band = 'r'
        tilename = 'DES0417-5914'
        os.environ['MEDS_DIR'] = str(tmpdir)
        medsconf = 'test_des_coadd_data'
        coaddsrcs = DESCoaddSources(
            medsconf=medsconf, tilename=tilename, band=band,
            campaign='Y3A1_COADD')
        coadd = DESCoadd(
            medsconf=medsconf, tilename=tilename, band=band,
            campaign='Y3A1_COADD', sources=coaddsrcs)

        info = coadd.get_info()
        assert 'src_info' in info

        # now try the download
        coadd.download()
        files = [
            os.path.join(d, f)
            for d, _, fs in os.walk(coadd.source_dir)
            for f in fs]
        assert len(files) > 1

        coadd.clean()
        assert not os.path.exists(coadd.source_dir)
    except Exception:
        os.environ.pop('MEDS_DIR', None)
        raise
