import numpy as np


def compute_unmasked_trail_fraction(*, bmask):
    """Compute the fraction of an image that has bleed trails marked outside
    of star masks.

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


def slice_full_edge_masked(*, weight, bmask, bad_flags):
    """Test if a given slice of an SE image has a full edge masked.

    We cannot interpolate with fully masked edges, so we should not use
    those images.

    Parameters
    ----------
    weight : array-like
        The weight map for the slice.
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        The flags to test in the bit mask using `(bmask & bad_flags) != 0`.

    Returns
    -------
    masked : bool
        `True` if a full edge is masked, `False` otherwise.
    """
    return (
        np.all(weight[0, :] <= 0.0) or
        np.all(weight[-1, :] <= 0.0) or
        np.all(weight[:, 0] <= 0.0) or
        np.all(weight[:, -1] <= 0.0) or
        np.all((bmask[0, :] & bad_flags) != 0) or
        np.all((bmask[-1, :] & bad_flags) != 0) or
        np.all((bmask[:, 0] & bad_flags) != 0) or
        np.all((bmask[:, -1] & bad_flags) != 0))


def slice_has_flags(*, bmask, flags):
    """Test if a slice has flags set.

    Parameters
    ----------
    bmask : array-like
        The bit mask for the slice.
    flags : int
        The flags to test in the bit mask using `(bmask & flags) != 0`.

    Returns
    -------
    edge : bool
        `True` if the slice has the flags set, `False` otherwise.
    """
    return np.any((bmask & flags) != 0)


def compute_masked_fraction(*, weight, bmask, bad_flags):
    """Compute the fraction of an image that is masked.

    Parameters
    ----------
    weight : array-like
        The weight map for the slice.
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        The flags to test in the bit mask using `(bmask & bad_flags) != 0`.

    Returns
    -------
    masked : float
        The fraction masked.
    """
    return np.mean((weight <= 0.0) | ((bmask & bad_flags) != 0))
