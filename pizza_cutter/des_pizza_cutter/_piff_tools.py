import os
from functools import lru_cache
import logging
import numpy as np
import piff
import fitsio
from ._constants import PSF_IN_BLACKLIST

logger = logging.getLogger(__name__)

def load_piff_from_image_path(*, image_path, piff_run):
    """
    load a piff object based on the input image path

    Parameters
    ----------
    image_path: str
        A path to an immask file
    piff_run: str
        e.g. y3a1-v29

    Returns
    -------
    A dict 
        {'psf': piff.PSF,
         'flags': int,
         'psf_path': str}
    """

    paths = _get_paths_from_image_path(image_path, piff_run)

    expinfo = _get_info(paths['info_path'])
    expnum, ccdnum = _extract_expnum_and_ccdnum(image_path)

    w,=np.where(expinfo['ccdnum'] == ccdnum)
    if w.size == 0:
        raise RuntimeError("piff info for exp %s ccd %s not found" % (expnum,ccdnum))

    this_info = expinfo[w[0]]

    piff_flags=0
    if not _check_and_log(this_info):
        piff_flags |= PSF_IN_BLACKLIST
        psf = None
    else:
        psf = _get_piff_psf(paths['psf_path'])

    return {
        'psf':psf,
        'flags':piff_flags,
        'psf_path': paths['psf_path'],
    }

def _check_and_log(info):
    """
    check the exp/ccd are OK based on flags and always skipping
    ccd 31
    """
    expnum, ccdnum = info['expnum'], info['ccdnum']
    ok=True
    if info['flag'] != 0:
        logger.info('skipping bad psf solution for exp %s '
                    'ccd %s: %s' % (expnum,ccd,info['flag']))
        ok=False

    if info['ccdnum'] == 31:
        logger.info('skipping ccd 31 for exp %s' % expnum)
        ok=False

    return ok

@lru_cache(maxsize=128)
def _get_piff_psf(psf_path):
    """
    load a piff.PSF object from the specified file
    """
    logger.info('reading: %s' % psf_path)
    return piff.read(psf_path)

@lru_cache(maxsize=128)
def _get_info(info_path):
    """
    read the info extension of the summary file
    """
    if not os.path.exists(info_path):
        raise RuntimeError('missing piff info file: %s' % info_path)

    return fitsio.read(info_path, ext='info')


def _extract_expnum_and_ccdnum(image_path):
    """
    extract the ccdnum from a path such as
    .../D00365173_i_c29_r2166p01_immasked.fits.fz
    """
    bname = os.path.basename(image_path)
    bs = bname.split('_')
    expnum = int(bs[0][1:])
    ccdnum = int( bs[2][1:] )
    return expnum, ccdnum

def _get_paths_from_image_path(image_path, piff_run):
    """
    get the piff and info path from the image path

    requires PIFF_DATA_DIR environment variable to be set

    Parameters
    ----------
    image_path: str
        A path to an immask file
    piff_run: str
        e.g. y3a1-v29

    Returns
    -------
    A dict with keys info_path and psf_path
    """
    PIFF_DATA_DIR = os.environ['PIFF_DATA_DIR']

    img_bname = os.path.basename(image_path)
    piff_bname = img_bname.replace('immasked.fits.fz', 'piff.fits')
    expnum = int(piff_bname.split('_')[0][1:])

    exp_dir = os.path.join(
        PIFF_DATA_DIR,
        piff_run,
        str(expnum),
    )

    psf_path = os.path.join(
        exp_dir,
        piff_bname,
    )
    info_path = os.path.join(
        exp_dir,
        'exp_psf_cat_%s.fits' % expnum,
    )

    return {
        'info_path': info_path,
        'psf_path': psf_path,
    }
