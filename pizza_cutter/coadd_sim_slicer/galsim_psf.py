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
    eval_locals : dict, optional
        An extra dictionary of local information to be used by `eval` when
        building PSFs from galsim eval-like strings.

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
    def __init__(self, psf, wcs, seed=1, method='auto', npix=33,
                 eval_locals=None):
        self.wcs = wcs
        self.method = method
        self.npix = npix
        self.seed = seed
        self._rng = np.random.RandomState(seed=self.seed)
        self.eval_locals = eval_locals

        if isinstance(psf, dict):
            self.psf_dict = psf
        else:
            self.psf = psf

    def _get_psf(self, row, col):
        if self.eval_locals is None:
            eval_locals = locals()
        else:
            eval_locals = copy.deepcopy(self.eval_locals)
            eval_locals['row'] = row
            eval_locals['col'] = col

        dct = copy.deepcopy(self.psf_dict)
        # hacking in some galsim eval stuff here...
        for k in dct:
            if k == 'type':
                continue
            elif isinstance(dct[k], str) and dct[k][0] == '$':
                dct[k] = eval(dct[k][1:], globals(), eval_locals)
        _psf, safe = galsim.config.BuildGSObject({'blah': dct}, 'blah')
        return _psf

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

        if not hasattr(self, 'psf'):
            psf = self._get_psf(row, col)
        else:
            psf = self.psf

        im = psf.drawImage(
            nx=self.npix,
            ny=self.npix,
            wcs=wcs,
            method=self.method).array.copy()
        if not isinstance(psf, galsim.InterpolatedImage):
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
        if not hasattr(self, 'psf'):
            psf = self._get_psf(row, col)
        else:
            psf = self.psf

        if not hasattr(psf, 'fwhm'):
            return measure_fwhm(self.get_rec(row, col)) / 2.35482004503
        else:
            return psf.fwhm / ps / 2.35482004503


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
        if psf is not None:
            self._psf_obj = galsim.des.des_psfex.DES_PSFEx(psf)

    def copy(self):
        new = GalSimPSFEx(None, npix=self.npix)
        new.psf = self.psf
        # trying a deep copy here to avoid refs to the orig meds file
        new._psf_obj = copy.deepcopy(self._psf_obj)
        return new

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
            method='no_pixel').array
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
