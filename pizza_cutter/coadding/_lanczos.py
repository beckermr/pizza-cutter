import numpy as np
import math
from numba import njit


@njit(fastmath=True)
def sinc(x):
    if abs(x) < 1e-4:
        return 1 - (math.pi * math.pi / 6) * x * x
    else:
        ax = math.pi * x
        return np.sin(ax) / ax


@njit(fastmath=True)
def lanczos_resample(im, rows, cols, a=3):
    """Lanczos resample one image at the input row and column positions.

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
    values, edge : np.ndarray
        The resampled value for each row, column pair. Points whose
        interpolation kernal was truncated because it extended beyond
        the input image have edge=True
    """

    ny, nx = im.shape
    outsize = rows.shape[0]

    res1 = np.zeros(outsize, dtype=np.float64)
    edge = np.zeros(outsize, dtype=np.bool_)

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
            x_s > nx-1 or
            y_f < 0 or
            y_s > ny-1
        )

        if out_of_bounds:
            edge[i] = True

        # now sum over the cells in the kernel
        val1 = 0.0
        for y_pix in range(y_s, y_f+1):
            if y_pix < 0 or y_pix > ny-1:
                continue

            dy = y - y_pix
            sy = sinc(dy) * sinc(dy/a)

            for x_pix in range(x_s, x_f+1):
                if x_pix < 0 or x_pix > nx-1:
                    continue

                dx = x - x_pix
                sx = sinc(dx) * sinc(dx/a)

                kernel = sx*sy

                val1 += im[y_pix, x_pix] * kernel

        res1[i] = val1

    return res1, edge


@njit(fastmath=True)
def lanczos_resample_two(im1, im2, rows, cols, a=3):
    """Lanczos resample two images at the input row and column positions.

    Unlike lanczos_resample, points whose interpolation kernel would be
    truncated because it extends beyond the input image do not get NaN values.
    Instead they get edge=True in the output edge boolean array

    Parameters
    ----------
    im1 : np.ndarray
        A two-dimensional array with the image values.
    im2 : np.ndarray
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
    values1, values2, edge : np.ndarray
        The resampled value for each row, column pair. Points whose
        interpolation kernal was truncated because it extended beyond
        the input image have edge=True
    """

    ny, nx = im1.shape
    outsize = rows.shape[0]

    res1 = np.zeros(outsize, dtype=np.float64)
    res2 = np.zeros(outsize, dtype=np.float64)
    edge = np.zeros(outsize, dtype=np.bool_)

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
            x_s > nx-1 or
            y_f < 0 or
            y_s > ny-1
        )

        if out_of_bounds:
            edge[i] = True

        # now sum over the cells in the kernel
        val1 = 0.0
        val2 = 0.0
        for y_pix in range(y_s, y_f+1):
            if y_pix < 0 or y_pix > ny-1:
                continue

            dy = y - y_pix
            sy = sinc(dy) * sinc(dy/a)

            for x_pix in range(x_s, x_f+1):
                if x_pix < 0 or x_pix > nx-1:
                    continue

                dx = x - x_pix
                sx = sinc(dx) * sinc(dx/a)

                kernel = sx*sy

                val1 += im1[y_pix, x_pix] * kernel
                val2 += im2[y_pix, x_pix] * kernel

        res1[i] = val1
        res2[i] = val2

    return res1, res2, edge
