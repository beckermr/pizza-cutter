import numpy as np
import fitsio


class BMaskGenerator(object):
    """A random generator for bit masks from an input catalog.

    Parameters
    ----------
    bmask_cat : str
        The path to the bit mask catalog. This should be a fits file with the
        bit masks stored in the 'msk' extension as a 1-d image. This image
        stores each bit mask is C-raveled order one afer the other. The fits
        file should have another binary table extension called 'metadata'
        which has the values:
            nrows : the number of rows for each bit mask
            ncols : the number of columns for each bit mask
    seed : int
        The seed for the RNG.

    Attributes
    ----------
    shape : tuple
        The output shape of the bit mask.

    Methods
    -------
    get_bmask(index)
        Get a bit mask for a specific index. Note that using the same seed
        and then requesting the same index will always return the same bit
        mask not matter what the call order.
    close()
        Close the underlying FITS object. Any calls to `get_bmask` aftre this
        will fail.
    """
    def __init__(self, *, bmask_cat, seed):
        self._seed = seed
        self._fits = fitsio.FITS(bmask_cat)
        m = self._fits['metadata'].read()
        self._nrows = m['nrows'][0]
        self._ncols = m['ncols'][0]
        self._npix = self._nrows * self._ncols
        self.shape = (self._nrows, self._ncols)
        self._n_masks = self._fits['msk'].get_dims()[0] // self._npix

    def get_bmask(self, index):
        """Get bit mask at random from the underlying catalog.

        Calls with the same seed and index will always return the same bit
        mask regardless of the order of the calls.

        Parameters
        ----------
        index : int
            The index of the mask to get.

        Returns
        -------
        bmask : np.array
            The bit mask.
        """
        _ind = np.random.RandomState(seed=self._seed).choice(
            index + self._n_masks)
        _ind = _ind % self._n_masks
        start = _ind * self._npix
        msk = self._fits['msk'][start:start + self._npix].copy()
        return msk.reshape(self._nrows, self._ncols)

    def close(self):
        """Close the underlying FITS object."""
        self._fits.close()
        delattr(self, '_fits')
