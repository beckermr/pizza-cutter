import copy

import numpy as np
import galsim
import galsim.des.des_psfex

from ..slice_utils.measure import measure_fwhm


class GalSimPSF(object):
    """A PSFEx-like wrapper for galsim objects to serve as PSFs.

    Based on a similar class by Niall (
        https://github.com/des-science/y3-wl_image_sims/blob/master/deblend/deblend_pipeline/steps.py#L422).

    Parameters
    ----------
    psf : dict or galsim object
        The specification of the PSF as a dictionary (i.e., the galsim yaml)
        or an actual galsim object.
    wcs : galsim WCS object
        The WCS for the PSF.
    method : string, optional
        The default method to use to draw the PSF. For observed PSFs, this
        should be 'no-pixel' otherwise the default of 'auto' is most likely
        correct.
    npix : int, optional
        The number of pixels for the PSF image. This number should be big
        enough to contain the full profile.
    seed : int, optional
        The random seed to use for adding noise to the PSF image.

    Methods
    -------
    get_rec(row, col)
        Get the PSF reconstruction at a point.
    get_center(row, col)
        Get the center of the PSF in the reconstruction.
    get_sigma(row, col)
        Get the PSF sigma at a point.

    Notes
    -----
    All input (row, col) are in zero-indexed, pixel-centered coordinates.
    """
    def __init__(self, psf, wcs, seed=1, method='auto', npix=33):
        self.wcs = wcs
        self.method = method
        self.npix = npix
        self.seed = seed
        self._rng = np.random.RandomState(seed=self.seed)

        if isinstance(psf, dict):
            self.psf_dict = psf
            _psf, safe = galsim.config.BuildGSObject(
                {'blah': copy.deepcopy(self.psf_dict)}, 'blah')
            if not safe:
                raise RuntimeError("The galsim object is not safe to reuse!")
            self.psf = _psf
        else:
            self.psf = psf

    def get_rec(self, row, col):
        """Get the PSF reconstruction at a point.

        All input (row, col) are in zero-indexed, pixel-centered coordinates.

        Parameters
        ----------
        row : float
            The row location of the desired center.
        col : float
            The col location of the desired center.

        Returns
        -------
        psf : np.array
            An image of the PSF.
        """
        im_pos = galsim.PositionD(col+1, row+1)
        wcs = self.wcs.local(im_pos)
        im = self.psf.drawImage(
            nx=self.npix,
            ny=self.npix,
            wcs=wcs,
            method=self.method).array.copy()
        im += self._rng.normal(scale=8e-4, size=im.shape)
        im /= im.sum()
        return im

    def get_center(self, row, col):
        """Get the center of the PSF in the reconstruction.

        All input (row, col) are in zero-indexed, pixel-centered coordinates.

        Parameters
        ----------
        row : float
            The row location of the desired center.
        col : float
            The col location of the desired center.

        Returns
        -------
        center : np.array
            The reconstruction center in zero-indexed, pixel-centered
            coordinates.
        """
        return np.array(((self.npix - 1) / 2, (self.npix - 1) / 2))

    def get_sigma(self, row, col):
        """Get the PSF sigma at a point.

        All input (row, col) are in zero-indexed, pixel-centered coordinates.

        Parameters
        ----------
        row : float
            The row location of the desired center.
        col : float
            The col location of the desired center.

        Returns
        -------
        sigma : float
            The PSF sigma in pixels.
        """
        ps = np.sqrt(self.wcs.pixelArea(
            image_pos=galsim.PositionD(col+1, row+1)))
        return self.psf.fwhm / ps / 2.35482004503


class GalSimPSFEx(object):
    """A PSFEx-like wrapper for galsim `DES_PSFEx` objects to serve as PSFs.

    Parameters
    ----------
    psf : str
        The path to the PSFEx reconstruction.
    npix : int, optional
        The number of pixels for the PSF image. This number should be big
        enough to contain the full profile.

    Methods
    -------
    get_rec(row, col)
        Get the PSF reconstruction at a point.
    get_center(row, col)
        Get the center of the PSF in the reconstruction.
    get_sigma(row, col)
        Get the PSF sigma at a point.

    Notes
    -----
    All input (row, col) are in zero-indexed, pixel-centered coordinates.
    """
    def __init__(self, psf, npix=33):
        self.npix = npix
        self.psf = psf
        self._psf_obj = galsim.des.des_psfex.DES_PSFEx(psf)

    def get_rec(self, row, col):
        """Get the PSF reconstruction at a point.

        All input (row, col) are in zero-indexed, pixel-centered coordinates.

        Parameters
        ----------
        row : float
            The row location of the desired center.
        col : float
            The col location of the desired center.

        Returns
        -------
        psf : np.array
            An image of the PSF.
        """
        im_pos = galsim.PositionD(col+1, row+1)
        im = self._psf_obj.getPSF(im_pos).drawImage(
            nx=self.npix,
            ny=self.npix,
            scale=1,
            method='nopixel').array.copy()
        im /= im.sum()
        return im

    def get_center(self, row, col):
        """Get the center of the PSF in the reconstruction.

        All input (row, col) are in zero-indexed, pixel-centered coordinates.

        Parameters
        ----------
        row : float
            The row location of the desired center.
        col : float
            The col location of the desired center.

        Returns
        -------
        center : np.array
            The reconstruction center in zero-indexed, pixel-centered
            coordinates.
        """
        return np.array(((self.npix - 1) / 2, (self.npix - 1) / 2))

    def get_sigma(self, row, col):
        """Get the PSF sigma at a point.

        All input (row, col) are in zero-indexed, pixel-centered coordinates.

        Parameters
        ----------
        row : float
            The row location of the desired center.
        col : float
            The col location of the desired center.

        Returns
        -------
        sigma : float
            The PSF sigma in pixels.
        """
        return measure_fwhm(self.get_rec(row, col)) / 2.35482004503
