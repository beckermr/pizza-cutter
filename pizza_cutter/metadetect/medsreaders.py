from ngmix.medsreaders import MultiBandNGMixMEDS as _MultiBandNGMixMEDS


class MultiBandNGMixMEDS(_MultiBandNGMixMEDS):
    """Interface to NGMixMEDS objects in more than one band.

    Parameters
    ----------
    mlist : list of `ngmix.medsreaders.NGMixMEDS` objects
        List of the `NGMixMEDS` objects for each band.

    Attributes
    ----------
    size : int
        The number of objects in the MEDS files.

    Methods
    -------
    get_mbobs_list(indices=None, weight_type='weight')
        Get a list of `MultiBandObsList` for all or a set of objects.
    get_mbobs(iobj, weight_type='weight')
        Get a `MultiBandObsList` for a given object.
    get_psf_rec_funcs(iobj)
        Get a list of functions to get the PSF at any point in the cutouts.
    get_wcs_jacobian_func(iobj)
        Get a function to return the WCS Jacobian at any point in the cutout.
    """

    def get_wcs_jacobian_func(self, iobj):
        """Get a function to return the WCS Jacobian at any point in the
        cutout.

        Parameters
        ----------
        iobj :
            Index of the object.

        Returns
        -------
        func : function
            A function with call signature `func(row, col)` that returns a
            dictionary containing the WCS Jacobian at the given input location.
        """
        return self.mlist[0].get_wcs_jacobian_func(iobj, 0)

    def get_psf_rec_funcs(self, iobj):
        """Get a list of functions that return the PSF image at a given row, col
        in each band

        Parameters
        ----------
        iobj : int
            Index of the object.

        Returns
        -------
        func_list : list of functions
            A list of functions with call signatures `func(row, col)` that
            return an image of the PSF at the given location for each band.
        """
        return [m.get_psf_rec_func(iobj, 0) for m in self.mlist]
