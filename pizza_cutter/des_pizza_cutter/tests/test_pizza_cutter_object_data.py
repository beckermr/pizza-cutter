import numpy as np

from .._pizza_cutter import _build_object_data


def test_pizza_cutter_build_object_data(coadd_image_data):
    d = _build_object_data(
        central_size=250,
        buffer_size=125,
        image_width=10000,
        wcs=coadd_image_data['eu_wcs'],
        psf_box_size=21,
        position_offset=coadd_image_data['position_offset']
    )

    assert np.array_equal(d['id'], np.arange(d.shape[0]))
    assert np.all(d['box_size'] == 500)
    assert np.all(d['file_id'] == 0)
    assert np.all(d['psf_box_size'] == 21)
    assert np.all(d['psf_cutout_row'] == 10)
    assert np.all(d['psf_cutout_col'] == 10)

    n = (10000 - 2*12) // 250
    half = (250 - 1) / 2
    for row_ind in range(n):
        for col_ind in range(n):
            row = row_ind * 250 + 125 + half
            col = col_ind * 250 + 125 + half
            index = row_ind * n + col_ind
            assert d['orig_row'][index, 0] == row
            assert d['orig_col'][index, 0] == col

            ra, dec = coadd_image_data['eu_wcs'].image2sky(
                col + coadd_image_data['position_offset'],
                row + coadd_image_data['position_offset']
            )

            assert d['ra'][index] == ra
            assert d['dec'][index] == dec

            assert d['orig_start_row'][index, 0] == row - 125 - half
            assert d['orig_start_col'][index, 0] == col - 125 - half
            assert d['cutout_row'][index, 0] == 125 + half
            assert d['cutout_col'][index, 0] == 125 + half

            ra, dec = coadd_image_data['eu_wcs'].image2sky(
                (d['orig_start_col'][index, 0] +
                 d['cutout_col'][index, 0] +
                 coadd_image_data['position_offset']),
                (d['orig_start_row'][index, 0] +
                 d['cutout_row'][index, 0] +
                 coadd_image_data['position_offset'])
            )

            assert d['ra'][index] == ra
            assert d['dec'][index] == dec

            jacob = coadd_image_data['eu_wcs'].get_jacobian(
                col + coadd_image_data['position_offset'],
                row + coadd_image_data['position_offset']
            )
            assert d['dudcol'][index, 0] == jacob[0]
            assert d['dudrow'][index, 0] == jacob[1]
            assert d['dvdcol'][index, 0] == jacob[2]
            assert d['dvdrow'][index, 0] == jacob[3]
