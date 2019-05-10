import numpy as np

from .._pizza_cutter import _build_image_info


def test_pizza_cutter_build_image_info():
    info = {
        'image_flags': 10,
        'image_id': 0,
        'magzp': 32,
        'scale': 0.5,
        'position_offset': 3,
        'wcs': "{'key1': 'val1'}",
        'src_info': [
            {
                'image_id': 1,
                'image_path': 'a',
                'image_ext': 'b',
                'weight_path': 'c',
                'weight_ext': 'd',
                'bmask_path': 'e',
                'bmask_ext': 'f',
                'bkg_path': 'g',
                'bkg_ext': 'h',
                'image_flags': 11,
                'magzp': 31,
                'scale': 0.75,
                'position_offset': 2,
                'image_wcs': "{'key2': 'val2'}"
            },
            {
                'image_id': 2,
                'image_path': 'aa',
                'image_ext': 'bb',
                'weight_path': 'cc',
                'weight_ext': 'dd',
                'bmask_path': 'ee',
                'bmask_ext': 'ff',
                'bkg_path': 'gg',
                'bkg_ext': 'hh',
                'image_flags': 12,
                'magzp': 30,
                'scale': 0.6,
                'position_offset': -1,
                'image_wcs': "{'key3': 'val3'}"
            }]
    }

    ii = _build_image_info(info=info)
    assert ii['image_flags'][0] == 10
    assert np.allclose(ii['magzp'][0], 32)
    assert np.allclose(ii['scale'][0], 0.5)
    assert ii['position_offset'][0] == 3
    assert ii['wcs'][0] == b'{"key1": "val1"}'

    assert ii['image_flags'][1] == 11
    assert np.allclose(ii['magzp'][1], 31)
    assert np.allclose(ii['scale'][1], 0.75)
    assert ii['position_offset'][1] == 2
    assert ii['wcs'][1] == b'{"key2": "val2"}'
    assert ii['image_path'][1] == b'a'
    assert ii['image_ext'][1] == b'b'
    assert ii['weight_path'][1] == b'c'
    assert ii['weight_ext'][1] == b'd'
    assert ii['bmask_path'][1] == b'e'
    assert ii['bmask_ext'][1] == b'f'
    assert ii['bkg_path'][1] == b'g'
    assert ii['bkg_ext'][1] == b'h'

    assert ii['image_flags'][2] == 12
    assert np.allclose(ii['magzp'][2], 30)
    assert np.allclose(ii['scale'][2], 0.6)
    assert ii['position_offset'][2] == -1
    assert ii['wcs'][2] == b'{"key3": "val3"}'
    assert ii['image_path'][2] == b'aa'
    assert ii['image_ext'][2] == b'bb'
    assert ii['weight_path'][2] == b'cc'
    assert ii['weight_ext'][2] == b'dd'
    assert ii['bmask_path'][2] == b'ee'
    assert ii['bmask_ext'][2] == b'ff'
    assert ii['bkg_path'][2] == b'gg'
    assert ii['bkg_ext'][2] == b'hh'
