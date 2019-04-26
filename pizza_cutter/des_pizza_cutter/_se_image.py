import numpy as np
import esutil as eu
import galsim
# import copy
#
# from ..memmappednoise import MemMappedNoiseImage


class SEImageSlice(object):
    """A single-epoch image w/ associated metadata from the DES.

    NOTE: All pixel coordinates are with respect to the original image,
    not the coordinates in the slice. By convetion all pixel coordinates
    for this class are pixel-centered and **zero-indexed**. This convetion
    does not match the usual one-indexed coordinate convetion in galsim
    and the DES data products. Appropriate conversions are made as needed.

    Parameters
    ----------
    source_info : dict
        The single-epoch source info from the `desmeds` classes.
    psf_model : a `galsim.GSObject`, `psfex.PSFEx`, `galsim.des.DES_PSFEx`,
            or `piff.PSF` object
        The PSF model to use. The type of input will be detected and then
        called appropriately.
    wcs : a ` esutil.wcsutil.WCS` or `galsim.BaseWCS` instance
        The WCS model to use.
    random_state : int, None, or np.random.RandomState, optional
        A random state to use. If None, a new RNG will be instantiated.

    Methods
    -------
    set_slice(x_start, y_start, box_size)
        Set the slice location and read a square slice of the
        image, weight map, bit mask, and the noise field.
    image2sky(x, y)
        Compute ra, dec for a given set of pixel coordinates.
    sky2image(ra, dec)
        Return x, y pixel coordinates for a given ra, dec.
    contains_radec(ra, dec)
        Use the approximate sky bounds to detect if the given point in
        ra, dec is anywhere in the total SE image (not just the slice).
    get_psf_image(x, y)
        Get an image of the PSF as a numpy array at the given location.
    get_psf_gsobject(x, y)
        Get the PSF as a `galsim.InterpolatedImage` with the appropriate
        WCS set at the given location.

    Attributes
    ----------
    source_info : dict
        The source info dictionary for the SE image.
    image : np.ndarray
        The slice of the image.
    weight : np.ndarray
        The slice of the wight map.
    bmask : np.ndarray
        The slice of the bit mask.
    noise : np.ndarray
        The slice of the noise field for the image.
    x_start : int
        The zero-indexed starting column for the slice.
    y_start : int
        The zero-indexed starting row for the slice.
    box_size : int
        The size of the slice.
    """
    def __init__(self, *, source_info, psf_model, wcs, random_state):
        self.source_info = source_info
        self._psf_model = psf_model
        self._wcs = wcs

        if isinstance(random_state, np.random.RandomState):
            self._rng = random_state
        else:
            self._rng = np.random.RandomState(seed=random_state)

    def set_slice(self, x_start, y_start, box_size):
        """Set the slice location and read a square slice of the
        image, weight map, bit mask, and the noise field."""
        raise NotImplementedError()

    def image2sky(self, x, y):
        """Compute ra, dec for a given set of pixel coordinates.

        Parameters
        ----------
        x : scalar or 1d array-like
            The x/column image location in zero-indexed, pixel centered
            coordinates.
        y : scalar or 1d array-like
            The y/row image location in zero-indexed, pixel centered
            coordinates.

        Returns
        -------
        ra : scalar or 1d array-like
            The right ascension of the sky position.
        dec : scalar or 1d array-like
            The declination of the sky position.
        """
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            is_scalar = True
        else:
            is_scalar = False

        assert np.ndim(x) <= 1 and np.ndim(y) <= 1, (
            "Inputs to image2sky must be scalars or 1d arrays")
        assert np.ndim(x) == np.ndim(y), (
            "Inputs to image2sky must be the same shape")

        x = np.atleast_1d(x).ravel()
        y = np.atleast_1d(y).ravel()

        if isinstance(self._wcs, eu.wcsutil.WCS):
            # for the DES we always have a one-indexed system
            ra, dec = self._wcs.image2sky(x+1, y+1)
        elif isinstance(self._wcs, galsim.BaseWCS):
            assert self._wcs.isCelestial()
            ra = []
            dec = []
            for _x, _y in zip(x, y):
                # for the DES we always have a one-indexed system
                image_pos = galsim.PositionD(x=_x+1, y=_y+1)
                world_pos = self._wcs.toWorld(image_pos)
                _ra = world_pos.ra / galsim.degrees
                _dec = world_pos.dec / galsim.degrees
                ra.append(_ra)
                dec.append(_dec)
            ra = np.array(ra)
            dec = np.array(dec)
        else:
            raise ValueError('WCS %s not recognized!' % self._wcs)

        if is_scalar:
            return ra[0], dec[0]
        else:
            return ra, dec

    def sky2image(self, ra, dec):
        """Return x, y image coordinates for a given ra, dec.

        Parameters
        ----------
        ra : scalar or 1d array-like
            The right ascension of the sky position.
        dec : scalar or 1d array-like
            The declination of the sky position.

        Returns
        -------
        x : scalar or 1d array-like
            The x/column image location in zero-indexed, pixel centered
            coordinates.
        y : scalar or 1d array-like
            The y/row image location in zero-indexed, pixel centered
            coordinates.
        """
        if np.ndim(ra) == 0 and np.ndim(dec) == 0:
            is_scalar = True
        else:
            is_scalar = False

        assert np.ndim(ra) <= 1 and np.ndim(dec) <= 1, (
            "Inputs to sky2mage must be scalars or 1d arrays")
        assert np.ndim(ra) == np.ndim(dec), (
            "Inputs to sky2image must be the same shape")

        ra = np.atleast_1d(ra).ravel()
        dec = np.atleast_1d(dec).ravel()

        if isinstance(self._wcs, eu.wcsutil.WCS):
            # for the DES we always have a one-indexed system
            # so we subtract to get back to zero
            x, y = self._wcs.sky2image(ra, dec)
            x -= 1
            y -= 1
        elif isinstance(self._wcs, galsim.BaseWCS):
            assert self._wcs.isCelestial()

            x = []
            y = []
            for _ra, _dec in zip(ra, dec):
                # for the DES we always have a one-indexed system
                world_pos = galsim.CelestialCoord(
                    ra=_ra*galsim.degrees,
                    dec=_dec*galsim.degrees
                )
                image_pos = self._wcs.toImage(world_pos)
                x.append(image_pos.x - 1)
                y.append(image_pos.y - 1)
            x = np.array(x)
            y = np.array(y)
        else:
            raise ValueError('WCS %s not recognized!' % self._wcs)

        if is_scalar:
            return x[0], y[0]
        else:
            return x, y

    def contains_radec(self, ra, dec):
        """Use the approximate sky bounds to detect if the given point in
        ra, dec is anywhere in the total SE image (not just the slice)."""
        raise NotImplementedError()

    def get_psf_image(self, x, y):
        """Get an image of the PSF as a numpy array at the given location."""
        raise NotImplementedError()

    def get_psf_gsobject(self, x, y):
        """Get the PSF as a `galsim.InterpolatedImage` with the appropriate
        WCS set at the given location."""
        raise NotImplementedError()
