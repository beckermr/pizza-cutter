import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


class WCSInversionInterpolator(object):
    """Interpolator to quickly invert a WCS solution.

    Suppose you have some WCS transformation of the form

        x_in, y_in = wcs(x_out, y_out)

    Usually, this function is fast to compute (e.g., image to sky) but the
    inverse is expensive (i.e., sky to image). In the case that `wcs` above
    is fast to compute but the inverse is slow, then you can use this class
    to construct an approximate interpolant for the inverse via

        >>> x_in, y_in = wcs(x_out, y_out)
        >>> interp = WCSInversionInterpolator(x_in, y_in, x_out, y_out)

    Then calling `interp` at a given point will compute the approximate
    x_out, y_out for a given input

        >>> interp(a, b)
        a_out, b_out

    Parameters
    ----------
    x_in : float or np.ndarray
        The input x/column value at which to evaluate the interpolant. This
        corresponds to the output x/column value of the WCS you
        want to invert.
    y_in : float or np.ndarray
        The input y/row value at which to evaluate the interpolant. This
        corresponds to the output y/row value of the WCS you
        want to invert.
    x_out : float or np.ndarray
        The output x/column value to be interpolated. This
        corresponds to the input x/column value of the WCS you
        want to invert.
    y_out : float or np.ndarray
        The output y/row value to be interpolated. This
        corresponds to the input y/row value of the WCS you
        want to invert.

    Methods
    -------
    __call__(x, y)
        Compute the values (x_out, y_out) corresponding to the input (x, y).
    """
    def __init__(self, x_in, y_in, x_out, y_out):
        pts = np.stack([y_in, x_in]).T
        self._x_int = CloughTocher2DInterpolator(pts, x_out)
        self._y_int = CloughTocher2DInterpolator(pts, y_out)

    def __call__(self, x, y):
        """Compute the values (x_out, y_out) corresponding to the input (x, y).

        Parameters
        ----------
        x : float or array-like
            The x/column values at which to compute the output values.
        y : float or array-like
            The y/row values at which to compute the output values.

        Returns
        -------
        x_out : float or array-like
            The computed output x/column value.
        y_out : float or array-like
            The computed output y/row value.
        """
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            is_scaler = True
        else:
            is_scaler = False
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        pts = np.stack([y, x]).T
        if is_scaler:
            return self._x_int(pts)[0], self._y_int(pts)[0]
        else:
            return self._x_int(pts), self._y_int(pts)
