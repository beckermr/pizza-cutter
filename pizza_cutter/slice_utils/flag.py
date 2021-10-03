import numpy as np


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


def compute_masked_fraction(*, weight, bmask, bad_flags, ignore_mask=None):
    """Compute the fraction of an image that is masked.

    Parameters
    ----------
    weight : array-like
        The weight map for the slice.
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        The flags to test in the bit mask using `(bmask & bad_flags) != 0`.
    ignore_mask : array-like, optional
        If not None, then the masked fraction is computed from only pixels
        marked as False in this mask.

    Returns
    -------
    masked : float
        The fraction masked.
    """
    if ignore_mask is not None:
        if np.all(ignore_mask):
            return 1.0
        else:
            keep_mask = ~ignore_mask
            return np.sum(
                ((weight <= 0.0) | ((bmask & bad_flags) != 0)) & keep_mask
            ) / np.sum(keep_mask)
    else:
        return np.mean((weight <= 0.0) | ((bmask & bad_flags) != 0))
