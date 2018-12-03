import os
import subprocess
import numpy as np
import fitsio
import pytest

from ..maskgen import BMaskGenerator


@pytest.fixture(scope='session')
def data(tmpdir_factory):
    tmpdir = tmpdir_factory.getbasetemp()
    seed = 42
    rng = np.random.RandomState(seed=seed)
    nmasks = 79
    nrows = 12
    ncols = 45
    npix = nmasks * nrows * ncols
    output_path = os.path.join(tmpdir, 'msk.fits')

    msk_img = rng.randint(0, 2**30, size=npix).astype(np.int32)
    metadata = np.zeros(1, dtype=[('nrows', 'i4'), ('ncols', 'i4')])
    metadata['nrows'] = nrows
    metadata['ncols'] = ncols

    fpack_pars = {
        'FZQVALUE': 4,
        'FZTILE': "(10240,1)",
        'FZALGOR': "RICE_1",
        # preserve zeros, don't dither them
        'FZQMETHD': "SUBTRACTIVE_DITHER_2",
    }

    with fitsio.FITS(output_path, clobber=True, mode='rw') as fits:
        # create the header and make sure to write the fpack keys so that
        # fpack does the right thing
        fits.create_image_hdu(
            img=None,
            dtype='i4',
            dims=[npix],
            extname='msk',
            header=fpack_pars)
        fits[0].write_keys(fpack_pars, clean=False)
        fits[0].write(msk_img, start=0)
        fits.write(metadata, extname='metadata')

    # now we fpack - this will save a ton of space because fpacking
    # integers is very efficient
    try:
        cmd = 'fpack ' + output_path
        subprocess.check_call(cmd, shell=True)
    except Exception:
        pass

    return {
        'nmasks': nmasks,
        'nrows': nrows,
        'ncols': ncols,
        'npix': npix,
        'msk_img': msk_img,
        'msk_path': output_path}


def test_bmaskgenerator(data):
    seed = 10
    gen = BMaskGenerator(bmask_cat=data['msk_path'], seed=seed)
    assert gen.shape == (data['nrows'], data['ncols'])

    inds = np.random.RandomState(seed=seed+1).choice(
        data['nmasks'], size=data['nmasks'], replace=False)

    size = data['nrows'] * data['ncols']
    for i in inds:
        bmsk = gen.get_bmask(i)
        _ind = np.random.RandomState(seed=seed).choice(i + data['nmasks'])
        _ind = _ind % data['nmasks']
        start = _ind * size
        assert np.array_equal(
            bmsk,
            data['msk_img'][start:start+size].reshape(
                data['nrows'], data['ncols']))


def test_bmaskgenerator_rng_order(data):
    seed = 10
    gen = BMaskGenerator(bmask_cat=data['msk_path'], seed=seed)
    assert gen.shape == (data['nrows'], data['ncols'])

    inds1 = np.random.RandomState(seed=seed+1).choice(
        data['nmasks'], size=data['nmasks'], replace=False)
    inds2 = np.random.RandomState(seed=seed+2).choice(
        data['nmasks'], size=data['nmasks'], replace=False)
    res1 = {}
    res2 = {}
    for ind1, ind2 in zip(inds1, inds2):
        res1[ind1] = gen.get_bmask(ind1)
        res2[ind2] = gen.get_bmask(ind2)

    for ind in inds1:
        assert np.array_equal(res1[ind], res2[ind])


def test_bmaskgenerator_rng_seed(data):
    seed = 10
    gen1 = BMaskGenerator(bmask_cat=data['msk_path'], seed=seed)
    gen2 = BMaskGenerator(bmask_cat=data['msk_path'], seed=seed)

    inds = np.random.RandomState(seed=seed+1).choice(
        data['nmasks'], size=data['nmasks'], replace=False)
    for i in inds:
        assert np.array_equal(gen1.get_bmask(i), gen2.get_bmask(i))
