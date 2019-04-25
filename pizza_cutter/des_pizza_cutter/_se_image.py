import numpy as np
# import esutil as eu
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
    def __init__(self, source_info, psf_model, wcs, random_state):
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
        """Compute ra, dec for a given set of pixel coordinates."""
        raise NotImplementedError()

    def sky2image(self, ra, dec):
        """Return x, y pixel coordinates for a given ra, dec."""
        raise NotImplementedError()

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
