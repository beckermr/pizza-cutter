import os

import numpy as np
import esutil as eu
import fitsio
import meds
import piff
import pixmappy
import desmeds
import ngmix
import scipy

from .._pizza_cutter import _build_metadata
from .._constants import MAGZP_REF
from meds.maker import MEDS_FMT_VERSION
from ... import __version__


def test_pizza_cutter_build_metadata(monkeypatch):
    monkeypatch.setenv('MEDS_DIR', 'BLAH')
    monkeypatch.setenv('PIFF_DATA_DIR', 'BLAHH')
    monkeypatch.setenv('DESDATA', 'BLAHHH')
    config = 'blah blah blah'
    json_info = "tile info"
    metadata, json_info_image = _build_metadata(config=config, json_info=json_info)

    assert np.all(metadata['numpy_version'] == np.__version__.encode('ascii'))
    assert np.all(metadata['scipy_version'] == scipy.__version__.encode('ascii'))
    assert np.all(metadata['esutil_version'] == eu.__version__.encode('ascii'))
    assert np.all(metadata['ngmix_version'] == ngmix.__version__.encode('ascii'))
    assert np.all(
        metadata['fitsio_version'] == fitsio.__version__.encode('ascii'))
    assert np.all(metadata['meds_version'] == meds.__version__.encode('ascii'))
    assert np.all(metadata['piff_version'] == piff.__version__.encode('ascii'))
    assert np.all(
        metadata['pixmappy_version'] == pixmappy.__version__.encode('ascii'))
    assert np.all(
        metadata['desmeds_version'] == desmeds.__version__.encode('ascii'))
    assert np.all(
        metadata['pizza_cutter_version'] == __version__.encode('ascii'))
    assert np.all(metadata['config'] == config.encode('ascii'))
    assert np.all(metadata['magzp_ref'] == MAGZP_REF)
    assert np.all(
        metadata['meds_fmt_version'] == MEDS_FMT_VERSION.encode('ascii'))
    assert np.all(
        metadata['meds_dir'] == os.environ['MEDS_DIR'].encode('ascii'))
    assert np.all(
        metadata['piff_data_dir'] ==
        os.environ['PIFF_DATA_DIR'].encode('ascii'))
    assert np.all(
        metadata['desdata'] == os.environ['DESDATA'].encode('ascii'))

    assert np.array_equal(json_info_image, np.array(json_info.encode("ascii")))
