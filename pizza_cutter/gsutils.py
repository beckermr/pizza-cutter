import galsim


def get_gs_fits_wcs_from_dict(wcs_dict):
    """Get a galsim.GSFitsWCS from an imahe header.

    This function does some extra checking and munging to code around
    bugs i've seen or fixed or reported.

    Parameters
    ----------
    wcs_dict : dict
        A dictionary of key-value pairs from the image header.

    Returns
    -------
    wcs : galsim.GSFitsWCS
        The WCS.
    """
    wcs = galsim.FitsWCS(
        header={k.upper(): wcs_dict[k] for k in wcs_dict.keys() if k is not None}
    )
    assert isinstance(wcs, galsim.GSFitsWCS)
    return wcs
