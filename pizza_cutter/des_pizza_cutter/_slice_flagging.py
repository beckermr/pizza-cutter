import numpy as np


def compute_unmasked_trail_fraction(*, bmask):
    """Compute the fraction of an image that has bleed trails marked outside
    of star masks.

    This function only works for DES Y3+ data.

    Parameters
    ----------
    bmask : array-like
        The bit mask for the slice.

    Returns
    -------
    frac : float
        The fraction of the image.
    """
    trail_not_masked = ((bmask & 64) != 0) & ((bmask & 32) == 0)
    return np.mean(trail_not_masked)
