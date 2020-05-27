import numpy as np
import esutil

from .._wcs_utils import WCSInversionInterpolator, WCSScalarInterpolator

COADD_WCS = esutil.wcsutil.WCS({
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
    'cunit1': 'deg     ',
    'crval1': 320.688891,
    'crpix1': 5000.5,
    'cd1_1': -7.305555555556e-05,
    'cd1_2': 0.0,
    'ctype2': 'DEC--TAN',
    'cunit2': 'deg     ',
    'crval2': 0.016667,
    'crpix2': 5000.5,
    'cd2_1': 0.0,
    'cd2_2': 7.305555555556e-05,
    'exptime': 450.0,
    'gain': 19.93043199192,
    'saturate': 31274.22430892,
    'softname': 'SWarp   ',
    'softvers': '2.40.0  ',
    'softdate': '2016-09-19',
    'softauth': '2010-2012 IAP/CNRS/UPMC',
    'softinst': 'IAP  http://www.iap.fr',
    'author': 'unknown ',
    'origin': 'nid18189',
    'date': '2016-10-13T00:30:52',
    'combinet': 'WEIGHTED',
    'bunit': 'electrons',
    'filter': 'i DECam SDSS c0003 7835.0 1470.0',
    'band': 'i       ',
    'tilename': 'DES2122+0001',
    'tileid': 90126,
    'resampt1': 'LANCZOS3',
    'centert1': 'MANUAL  ',
    'pscalet1': 'MANUAL  ',
    'resampt2': 'LANCZOS3',
    'centert2': 'MANUAL  ',
    'pscalet2': 'MANUAL  ',
    'desfname': 'DES2122+0001_r2601p01_i.fits',
    'pipeline': 'multiepoch',
    'unitname': 'DES2122+0001',
    'attnum': 1,
    'eupsprod': 'MEPipeline',
    'eupsver': 'Y3A1+0  ',
    'reqnum': 2601,
    'des_ext': 'IMAGE   ',
    'fzalgor': 'RICE_1  ',
    'fzdthrsd': 'CHECKSUM',
    'fzqvalue': 16,
    'fzqmethd': 'SUBTRACTIVE_DITHER_2',
    'ra_cent': 320.688927527779,
    'dec_cent': 0.016630472222219,
    'rac1': 321.054126640963,
    'decc1': -0.348562220896466,
    'rac2': 320.323655359037,
    'decc2': -0.348562220896466,
    'rac3': 320.323654004521,
    'decc3': 0.381895543631536,
    'rac4': 321.054127995479,
    'decc4': 0.381895543631536,
    'racmin': 320.323654004521,
    'racmax': 321.054127995479,
    'deccmin': -0.348562220896466,
    'deccmax': 0.381895543631536,
    'crossra0': 'N       ',
    'magzero': 30.0,
    'history': "'SUBTRACTIVE_DITHER_2' / Pixel Quantization Algorithm",
    'zquantiz': 'SUBTRACTIVE_DITHER_2',
    'zdither0': 5591,
    'checksum': 'ZJHKaGGKUGGKZGGK',
    'datasum': 1452922543})

SE_WCS = esutil.wcsutil.WCS({
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
    'extname': 'SCI     ',
    'proctype': 'RAW     ',
    'prodtype': 'image   ',
    'pixscal1': 0.27,
    'pixscal2': 0.27,
    'obs-long': 70.81489,
    'telescop': 'CTIO 4.0-m telescope',
    'observat': 'CTIO    ',
    'obs-lat': -30.16606,
    'obs-elev': 2215.0,
    'instrume': 'DECam   ',
    'expreq': 90.0,
    'exptime': 90.0,
    'darktime': 91.1224601,
    'obsid': 'ct4m20130910t034817',
    'date-obs': '2013-09-10T03:48:17.031944',
    'time-obs': '03:48:17.031944',
    'mjd-obs': 56545.15853046,
    'openshut': '2013-09-10T03:48:17.084340',
    'timesys': 'UTC     ',
    'expnum': 232321,
    'object': 'DES survey hex -395+6 tiling 1',
    'obstype': 'raw_obj ',
    'camshut': 'Open    ',
    'program': 'survey  ',
    'observer': 'TE, ES  ',
    'proposer': 'Frieman ',
    'dtpi': 'Frieman           ',
    'propid': '2012B-0001',
    'excluded': '        ',
    'hex': 4235,
    'tiling': 1,
    'seqid': '-395+6 enqueued on 2013-09-10 03:30:33Z by SurveyTactician',
    'seqnum': 1,
    'seqtot': 1,
    'aos': True,
    'bcam': True,
    'guider': 1,
    'skystat': True,
    'filter': 'i DECam SDSS c0003 7835.0 1470.0',
    'filtpos': 'cassette_1',
    'instance': 'DECam_20130909',
    'errors': None,
    'telequin': 2000.0,
    'telstat': 'Track   ',
    'ra': '21:21:57.696',
    'dec': '00:36:52.200',
    'telra': '21:21:57.697',
    'teldec': '00:36:48.100',
    'ha': '00:59:45.460',
    'zd': 33.95,
    'az': 332.6,
    'telfocus': (1479.03, -2947.54, 1803.5, 127.89, -57.91, -0.0),
    'vsub': True,
    'gskyphot': False,
    'lskyphot': True,
    'windspd': 12.231,
    'winddir': 179.0,
    'humidity': 15.0,
    'pressure': 781.0,
    'dimmsee': 0.617,
    'outtemp': 18.3,
    'airmass': 1.21,
    'gskyvar': 0.05,
    'gskyhot': 0.18,
    'lskyvar': 0.01,
    'lskyhot': 0.0,
    'lskypow': 0.01,
    'msurtemp': 17.875,
    'mairtemp': 18.3,
    'uptrtemp': 19.589,
    'lwtrtemp': -999.0,
    'pmostemp': 18.3,
    'utn-temp': 19.48,
    'uts-temp': 19.545,
    'utw-temp': 19.8,
    'ute-temp': 19.53,
    'pmn-temp': 17.2,
    'pms-temp': 17.8,
    'pmw-temp': 18.7,
    'pme-temp': 17.8,
    'domelow': 19.465,
    'domehigh': -999.0,
    'domeflor': 17.5,
    'g-meanx': 0.0463,
    'g-meany': 0.0632,
    'donutfs4': [0.64, 1.17, -8.86, -0.5, 0.11, 0.07, 0.46, 0.0, -0.48],
    'donutfs3': [-0.21, 0.93, 8.75, -0.5, 0.23, 0.06, 0.3, 0.2, -0.59],
    'donutfs2': [0.11, 0.65, -9.73, -0.19, 0.11, -0.02, 0.3, 0.35, -0.1],
    'donutfs1': [0.12, 1.25, 8.62, -0.34, 0.23, 0.18, 0.34, 0.34, -0.0],
    'g-flxvar': 18593751.61,
    'g-meanxy': -0.005538,
    'donutfn1': [-0.05, -0.2, -9.58, -0.26, 0.64, -0.09, 0.2, 0.47, -0.35],
    'donutfn2': [1.98, 0.88, 8.96, -0.87, -0.33, 0.0, 0.07, 0.35, -0.13],
    'donutfn3': [0.46, 0.52, -9.07, -0.39, -0.34, 0.12, 0.07, 0.15, 0.18],
    'donutfn4': [-1.54, -0.2, 8.42, -0.3, -0.86, 0.15, 0.17, 0.16, -0.11],
    'time_recorded': '2013-09-10T03:50:09.207158',
    'g-feedbk': 10,
    'g-ccdnum': 4,
    'doxt': 0.02,
    'g-maxx': 0.2885,
    'fadz': 53.82,
    'fady': 275.96,
    'fadx': 262.36,
    'g-mode': 'auto    ',
    'fayt': -4.23,
    'dodz': 53.82,
    'dody': -0.19,
    'dodx': -0.63,
    'bcamaz': 0.0,
    'multiexp': False,
    'bcamax': -32.47,
    'bcamay': -12.147,
    'bcamdy': 1020.038,
    'bcamdx': -511.087,
    'skyupdat': '2013-09-10T03:46:31',
    'g-seeing': 1.985,
    'g-transp': 0.796,
    'g-meany2': 0.017566,
    'doyt': 0.42,
    'g-latenc': 1.308,
    'lutver': 'working-trunk',
    'faxt': -9.11,
    'g-maxy': 0.3037,
    'g-meanx2': 0.013485,
    'sispiver': 'trunk   ',
    'constver': 'DECAM:19',
    'hdrver': 13,
    'dtsite': 'ct                ',
    'dttelesc': 'ct4m              ',
    'dtinstru': 'decam             ',
    'dtcaldat': '2013-09-09        ',
    'odateobs': '                  ',
    'dtutc': '2013-09-10T03:50:41',
    'dtobserv': 'NOAO              ',
    'dtpropid': '2012B-0001        ',
    'dtpiaffl': '                  ',
    'dttitle': '                  ',
    'dtcopyri': 'AURA              ',
    'dtacquis': 'pipeline3.ctio.noao.edu',
    'dtaccoun': 'sispi             ',
    'dtacqnam': '/data_local/images/DTS/2012B-0001/DECam_00232321.fits.fz',
    'dtnsanam': 'dec103900.fits    ',
    'dtqueue': 'des               ',
    'dtstatus': 'done              ',
    'sb_host': 'pipeline3.ctio.noao.edu',
    'sb_accou': 'sispi             ',
    'sb_site': 'ct                ',
    'sb_local': 'dec               ',
    'sb_dir1': 20130909,
    'sb_dir2': 'ct4m              ',
    'sb_dir3': '2012B-0001        ',
    'sb_recno': 103900,
    'sb_id': 'dec103900         ',
    'sb_name': 'dec103900.fits    ',
    'rmcount': 0,
    'recno': 103900,
    'bunit': 'electrons',
    'wcsaxes': 2,
    'detsize': '[1:29400,1:29050]',
    'datasec': '[1:2048,1:4096]',
    'detsec': '[22529:24576,10240:14335]',
    'ccdsec': '[1:2048,1:4096]',
    'detseca': '[22529:23552,10240:14335]',
    'ccdseca': '[1:1024,1:4096]',
    'ampseca': '[1:1024,1:4096]',
    'dataseca': '[1:1024,1:4096]',
    'detsecb': '[23553:24576,10240:14335]',
    'ccdsecb': '[1025:2048,1:4096]',
    'ampsecb': '[2048:1025,1:4096]',
    'datasecb': '[1025:2048,1:4096]',
    'detector': 'S3-119_123194-11-3',
    'ccdnum': 57,
    'detpos': 'N26     ',
    'gaina': 1.00274243961024,
    'rdnoisea': 5.932,
    'saturata': 137491.780096181,
    'gainb': 0.983317669448972,
    'rdnoiseb': 5.725,
    'saturatb': 123794.490612799,
    'crpix1': -9120.8,
    'crpix2': 4177.667,
    'fpa': 'DECAM_BKP5',
    'ccdbin1': 1,
    'ccdbin2': 1,
    'dheinf': 'MNSN fermi hardware',
    'dhefirm': 'demo30  ',
    'slot00': 'MCB 7 5.210000',
    'slot01': 'DESCB 23 4.010000',
    'slot02': 'DESCB 0x0 4.010000',
    'slot03': 'CCD12 3 4.080000',
    'slot04': 'CCD12 23 4.080000',
    'slot05': 'CCD12 13 4.080000',
    'radesys': 'ICRS    ',
    'equinox': 2000.0,
    'pv1_7': -0.001131856392163,
    'cunit1': 'deg     ',
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
    'cunit2': 'deg     ',
    'cd2_1': -7.285403390983e-05,
    'cd2_2': -1.476988018249e-07,
    'ltv2': 0.0,
    'ltv1': 0.0,
    'pv1_10': -0.006122290458248,
    'ctype2': 'DEC--TPV',
    'ctype1': 'RA---TPV',
    'crval1': 320.4912462427,
    'crval2': 0.6171111312777,
    'valida': True,
    'validb': True,
    'ndonuts': 0,
    '': '',
    'photflag': 1,
    'desdcxtk': 'Thu Mar 31 15:15:52 2016',
    'xtalkfil': 'DECam_20130606.xtalk',
    'desoscn': 'Thu Mar 31 15:15:52 2016',
    'fzalgor': 'RICE_1  ',
    'fzqmethd': 'SUBTRACTIVE_DITHER_2',
    'fzqvalue': 16,
    'fzdthrsd': 'CHECKSUM',
    'band': 'i       ',
    'camsym': 'D       ',
    'nite': 20130909,
    'desfname': 'D00232321_i_c57_r2357p01_immasked.fits',
    'pipeline': 'finalcut',
    'unitname': 'D00232321',
    'attnum': 1,
    'eupsprod': 'finalcut',
    'eupsver': 'Y2A1+5  ',
    'reqnum': 2357,
    'biasfil': 'D_n20130916t0926_c57_r1999p06_biascor.fits',
    'desbias': 'Thu Mar 31 15:34:10 2016',
    'lincfil': 'lin_tbl_v0.4.fits',
    'deslinc': 'Thu Mar 31 15:34:12 2016',
    'desbpm': 'Thu Mar 31 15:34:13 2016',
    'bpmfil': 'D_n20130916t0926_c57_r2083p01_bpm.fits',
    'dessat': 'Thu Mar 31 15:34:13 2016',
    'nsatpix': 13261,
    'flatmeda': 1.00274243961024,
    'flatmedb': 0.983317669448972,
    'saturate': 137491.780096181,
    'desgainc': 'Thu Mar 31 15:34:13 2016',
    'bfcfil': 'D_n20150305_r1428p01_bf.fits',
    'desbfc': 'Thu Mar 31 15:34:15 2016',
    'flatfil': 'D_n20130916t0926_i_c57_r1999p06_norm-dflatcor.fits',
    'desflat': 'Thu Mar 31 15:34:15 2016',
    'fixcfil': 'D_n20130916t0926_c57_r2083p01_bpm.fits',
    'desfixc': 'Thu Mar 31 15:34:15 2016',
    'ra_cent': 320.334208216425,
    'dec_cent': -0.122655583198652,
    'rac1': 320.184653593733,
    'decc1': -0.0476310761279146,
    'rac2': 320.184559081817,
    'decc2': -0.196824403983542,
    'rac3': 320.48379991099,
    'decc3': -0.197603840171634,
    'rac4': 320.484084276791,
    'decc4': -0.0484244712764058,
    'racmin': 320.184559081817,
    'racmax': 320.484084276791,
    'deccmin': -0.197603840171634,
    'deccmax': -0.0476310761279146,
    'crossra0': 'N       ',
    'fwhm': 3.8783,
    'scampchi': 2.8819,
    'elliptic': 0.0591,
    'scampnum': 1317,
    'scampref': 'UCAC-4  ',
    'desbleed': 'Thu Mar 31 16:16:51 2016',
    'nbleed': 45027,
    'starmask': 'Thu Mar 31 16:16:51 2016',
    'des_ext': 'IMAGE   ',
    'skysbfil': 'Y2A1_20130801t1128_i_c57_r2044p01_skypca-tmpl.fits',
    'skypc00': 2339.17986012236,
    'skypc01': -5.69772070194055,
    'skypc02': 10.3202256124756,
    'skypc03': -1.68147589700257,
    'skyvara': 2351.65820178295,
    'skyvarb': 2391.03565464325,
    'skysigma': 48.6973713882834,
    'skybrite': 2317.88702996854,
    'desskysb': 'Thu Mar 31 16:59:03 2016',
    'starfil': 'Y2A1_20130801t1128_i_c57_r2046p01_starflat.fits',
    'desstar': 'Thu Mar 31 17:59:13 2016',
    'desncray': 133,
    'desnstrk': 0,
    'desimmsk': 'Thu Mar 31 23:39:36 2016',
    'zquantiz': 'SUBTRACTIVE_DITHER_2',
    'history': "'SUBTRACTIVE_DITHER_2' / Pixel Quantization Algorithm",
    'zdither0': 3543,
    'checksum': '5SXRASVQ7SVQASVQ',
    'datasum': 1755117338})


def test_wcs_inversion():
    rng = np.random.RandomState(seed=10)
    dim = 50
    y_out, x_out = np.mgrid[:dim+2:2, 0:dim+2:2]
    y_out = y_out.ravel()
    x_out = x_out.ravel()
    x, y = COADD_WCS.sky2image(*SE_WCS.image2sky(x_out, y_out))

    wcs_inv = WCSInversionInterpolator(x, y, x_out, y_out)

    for _ in range(10):
        se_pos = rng.uniform(size=2)*49 + 1
        se_pos = (se_pos[0], se_pos[1])
        coadd_pos = COADD_WCS.sky2image(*SE_WCS.image2sky(*se_pos))
        inv_se_pos = SE_WCS.sky2image(*COADD_WCS.image2sky(*coadd_pos))
        interp_se_pos = wcs_inv(*coadd_pos)

        assert np.allclose(inv_se_pos, interp_se_pos)
        assert np.allclose(se_pos, interp_se_pos)


def test_wcs_scalar_interp():
    rng = np.random.RandomState(seed=10)
    dim = 50
    y, x = np.mgrid[:dim+2:2, 0:dim+2:2]
    y = y.ravel()
    x = x.ravel()
    tup = SE_WCS.get_jacobian(x, y)
    area = tup[0]**2

    wcs_area = WCSScalarInterpolator(x, y, area)

    for _ in range(10):
        se_pos = rng.uniform(size=2)*49 + 1
        _area = SE_WCS.get_jacobian(se_pos[0], se_pos[1])[0]**2
        interp_area = wcs_area(se_pos[0], se_pos[1])
        assert np.allclose(_area, interp_area)
