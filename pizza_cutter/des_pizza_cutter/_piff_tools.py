from functools import lru_cache
import piff


@lru_cache(maxsize=128)
def get_piff_psf(psf_path):
    """load a piff.PSF object from the specified file"""
    return piff.read(psf_path)
