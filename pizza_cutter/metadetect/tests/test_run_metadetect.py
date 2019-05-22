import numpy as np

from esutil.wcsutil import WCS

from ..run_metadetect import _make_output_array


def test_make_output_array():
    wcs = WCS(dict(
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
        crval2=1.444444))
    position_offset = 2
    orig_start_col = 10
    orig_start_row = 20
    slice_id = 11
    mcal_step = b'blah'

    dtype = [
        ('sx_row', 'f8'),
        ('sx_col', 'f8'),
        ('sx_row_noshear', 'f8'),
        ('sx_col_noshear', 'f8'),
        ('a', 'f8'),
    ]
    data = np.zeros(10, dtype=dtype)

    data['sx_row'] = np.arange(10) + 324
    data['sx_col'] = np.arange(10) + 3
    data['sx_row_noshear'] = np.arange(10) + 324
    data['sx_col_noshear'] = np.arange(10) + 3

    arr = _make_output_array(
        data=data,
        slice_id=slice_id,
        mcal_step=mcal_step,
        orig_start_row=orig_start_row,
        orig_start_col=orig_start_col,
        position_offset=position_offset,
        wcs=wcs)

    assert np.all(arr['slice_id'] == slice_id)
    assert np.all(arr['mcal_step'] == mcal_step)

    ra, dec = wcs.image2sky(
        x=data['sx_col'] + orig_start_col + position_offset,
        y=data['sx_row'] + orig_start_row + position_offset,
    )
    ura, udec = wcs.image2sky(
        x=data['sx_col_noshear'] + orig_start_col + position_offset,
        y=data['sx_row_noshear'] + orig_start_row + position_offset,
    )
    assert np.all(arr['ra'] == ra)
    assert np.all(arr['dec'] == dec)
    assert np.all(arr['ra_noshear'] == ura)
    assert np.all(arr['dec_noshear'] == udec)
