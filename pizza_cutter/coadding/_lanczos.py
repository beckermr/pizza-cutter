import numpy as np
from numba import njit


@njit
def lanczos_resample(im, rows, cols, a=3):
    """Lanczos resample an image at the input row and column positions.

    Parameters
    ----------
    im : np.ndarray
        A two-dimensional array with the image values.
    rows : np.ndarray
        A one-dimensional array of input row/y values. These denote the
        location to sample on the first, slowest moving axis of the image.
    cols : np.ndarray
        A one-dimensional array of input column/x values. These denote the
        location to sample on the second, fastest moving axis of the image.
    a : int, optional
        The size of the Lanczos kernel. The default of 3 is a good choice
        for many applications.

    Returns
    -------
    values : np.ndarray
        The resampled value for each row, column pair. Points whose
        interpolation kernal does not touch any part of the grid are
        returned as NaN.
    """
    res = np.zeros(rows.shape[0], dtype=np.float64)

    for i in range(rows.shape[0]):
        y = rows[i]
        x = cols[i]

        # get range for kernel
        x_s = int(np.floor(x)) - a + 1
        x_f = int(np.floor(x)) + a
        y_s = int(np.floor(y)) - a + 1
        y_f = int(np.floor(y)) + a

        out_of_bounds = (
            x_f < 0 or
            x_s > im.shape[1]-1 or
            y_f < 0 or
            y_s > im.shape[0]-1)
        if out_of_bounds:
            res[i] = np.nan
            continue

        # clip the kernel to the input image if needed
        x_s = max(0, min(x_s, im.shape[1]-1))
        x_f = max(0, min(x_f, im.shape[1]-1))
        y_s = max(0, min(y_s, im.shape[0]-1))
        y_f = max(0, min(y_f, im.shape[0]-1))

        # now sum over the cells in the kernel
        val = 0.0
        for y_pix in range(y_s, y_f+1):
            dy = y - y_pix
            for x_pix in range(x_s, x_f+1):
                dx = x - x_pix
                val += (
                    im[y_pix, x_pix] *
                    np.sinc(dx) * np.sinc(dx/a) *
                    np.sinc(dy) * np.sinc(dy/a))
        res[i] = val

    return res
