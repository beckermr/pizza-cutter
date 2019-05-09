import os
import pytest
import yaml
import fitsio

import galsim
import esutil as eu


@pytest.fixture
def se_image_data():
    se_wcs_data = {
        'xtension': 'BINTABLE',
        'bitpix': 8,
        'naxis': 2,
        'naxis1': 24,
        'naxis2': 4096,
        'pcount': 7424352,
        'gcount': 1,
        'tfields': 3,
        'ttype1': 'COMPRESSED_DATA',
        'tform1': '1PB(2305)',
        'ttype2': 'ZSCALE  ',
        'tform2': '1D      ',
        'ttype3': 'ZZERO   ',
        'tform3': '1D      ',
        'zimage': True,
        'ztile1': 2048,
        'ztile2': 1,
        'zcmptype': 'RICE_ONE',
        'zname1': 'BLOCKSIZE',
        'zval1': 32,
        'zname2': 'BYTEPIX ',
        'zval2': 4,
        'zsimple': True,
        'zbitpix': -32,
        'znaxis': 2,
        'znaxis1': 2048,
        'znaxis2': 4096,
        'zextend': True,
        'crpix1': -9120.8,
        'crpix2': 4177.667,
        'radesys': 'ICRS    ',
        'equinox': 2000.0,
        'pv1_7': -0.001131856392163,
        'cunit1': 'deg',
        'pv2_8': 0.001018303032252,
        'pv2_9': 0.002319394606743,
        'cd1_1': -1.48270437561e-07,
        'ltm2_2': 1.0,
        'ltm2_1': 0.0,
        'pv2_0': -0.003399720238217,
        'pv2_1': 0.9864515588353,
        'pv2_2': 0.0009454823496124,
        'pv2_3': 0.0,
        'pv2_4': -0.02314806967003,
        'pv2_5': 0.001877677471197,
        'pv2_6': 0.004309589780532,
        'pv2_7': -0.01227383889951,
        'ltm1_1': 1.0,
        'pv1_6': -0.01361136561823,
        'pv2_10': 0.0009498695718565,
        'pv1_4': 0.003530898113869,
        'pv1_3': 0.0,
        'pv1_2': -0.01014864986384,
        'pv1_1': 1.008025318525,
        'pv1_0': -0.002359709297272,
        'ltm1_2': 0.0,
        'pv1_9': 0.000779072746685,
        'pv1_8': 0.003705666166824,
        'cd1_2': 7.285803899392e-05,
        'pv1_5': 0.006384496695735,
        'cunit2': 'deg',
        'cd2_1': -7.285403390983e-05,
        'cd2_2': -1.476988018249e-07,
        'ltv2': 0.0,
        'ltv1': 0.0,
        'pv1_10': -0.006122290458248,
        'ctype2': 'DEC--TPV',
        'ctype1': 'RA---TPV',
        'crval1': 320.4912462427,
        'crval2': 0.6171111312777}

    if 'TEST_DESDATA' in os.environ:
        DESDATA = os.path.join(os.environ['TEST_DESDATA'], 'DESDATA')
        PIFF_DATA_DIR = os.path.join(
            os.environ['TEST_DESDATA'], 'PIFF_DATA_DIR')

        pth = os.path.join(
            os.environ['TEST_DESDATA'], 'source_info.yaml')
        with open(pth, 'r') as fp:
            source_info = yaml.load(fp, Loader=yaml.Loader)
        source_info['weight_path'] = source_info['image_path']
        source_info['bmask_path'] = source_info['image_path']

        for k in source_info:
            if '_path' in k and k != 'piff_path':
                source_info[k] = os.path.join(DESDATA, source_info[k])
        source_info['piff_path'] = os.path.join(
            PIFF_DATA_DIR, source_info['piff_path'])

        _se_wcs_data = fitsio.read_header(
            source_info['image_path'],
            ext=source_info['image_ext'])
        se_wcs_data = {
            k.lower(): _se_wcs_data[k] for k in _se_wcs_data.keys()}

        source_info['position_offset'] = 1.0
    else:
        source_info = None

    return {
        'source_info': source_info,
        'wcs_header': se_wcs_data,
        'eu_wcs': eu.wcsutil.WCS(se_wcs_data),
        'gs_wcs': galsim.FitsWCS(
            header={k.upper(): v for k, v in se_wcs_data.items()})
    }


@pytest.fixture
def coadd_image_data():
    wcs = {
        'xtension': 'BINTABLE',
        'bitpix': 8,
        'naxis': 2,
        'naxis1': 24,
        'naxis2': 10000,
        'pcount': 83807884,
        'gcount': 1,
        'tfields': 3,
        'ttype1': 'COMPRESSED_DATA',
        'tform1': '1PB(8590)',
        'ttype2': 'ZSCALE  ',
        'tform2': '1D      ',
        'ttype3': 'ZZERO   ',
        'tform3': '1D      ',
        'zimage': True,
        'ztile1': 10000,
        'ztile2': 1,
        'zcmptype': 'RICE_ONE',
        'zname1': 'BLOCKSIZE',
        'zval1': 32,
        'zname2': 'BYTEPIX ',
        'zval2': 4,
        'zsimple': True,
        'zbitpix': -32,
        'znaxis': 2,
        'znaxis1': 10000,
        'znaxis2': 10000,
        'zextend': True,
        'extname': 'SCI     ',
        'equinox': 2000.0,
        'mjd-obs': 56545.15853046,
        'radesys': 'ICRS    ',
        'ctype1': 'RA---TAN',
        'cunit1': 'deg',
        'crval1': 320.688891,
        'crpix1': 5000.5,
        'cd1_1': -7.305555555556e-05,
        'cd1_2': 0.0,
        'ctype2': 'DEC--TAN',
        'cunit2': 'deg',
        'crval2': 0.016667,
        'crpix2': 5000.5,
        'cd2_1': 0.0,
        'cd2_2': 7.305555555556e-05}

    return {
        'wcs_header': wcs,
        'eu_wcs': eu.wcsutil.WCS(wcs),
        'position_offset': 1
    }
