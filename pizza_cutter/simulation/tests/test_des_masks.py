import os

import numpy as np
import fitsio

import pytest

from ..des_masks import gen_masks_from_des_y3_images


@pytest.fixture(scope='session')
def data(tmpdir_factory):
    tmpdir = tmpdir_factory.getbasetemp()

    pths = []
    nmasks = 10
    for i in range(nmasks):
        pth = os.path.join(tmpdir, 'msk%02d.fits' % i)

        msk = np.zeros((512, 256), dtype=np.int32)

        # always ignored
        msk[:, :] = 1024

        if i == 0:
            # this sets every 20th pixel in each dim to a value we
            # exclude
            msk[::20, ::20] |= 64
        elif i == 1:
            # mask the whole thing
            msk |= 512
        elif i == 2:
            # we will test for this value
            msk[::16, ::16] = 16
        else:
            # all of these should be ignored
            msk[:, :] = 32 | 8 | 4

        fitsio.write(pth, msk)

        pths.append(pth)

    return {
        'bmask_paths': pths,
        'bmask_ext': 0,
        'output_path': os.path.join(tmpdir, 'mskcat.fits'),
        'n_per': 10,
        'seed': 42,
        'nrows': 200,
        'ncols': 100}


def test_gen_masks_from_des_y3_images(data):
    gen_masks_from_des_y3_images(**data)

    assert os.path.exists(data['output_path'] + '.fz')

    with fitsio.FITS(data['output_path'] + '.fz', mode='r') as fits:
        metadata = fits['metadata'].read()
        assert np.all(metadata['nrows'] == data['nrows'])
        assert np.all(metadata['ncols'] == data['ncols'])

        msk = fits['msk'].read()
        npix = msk.shape[0]
        nmasks = npix // data['nrows'] // data['ncols']

        # we lose 10 from bleed strails and 10 from too much area masked
        assert nmasks == 80

        # values 4 and 8 should always be ignored
        assert not np.any(msk & 4)
        assert not np.any(msk & 4)

        # some 16 values should be there
        assert np.any(msk & 16)
