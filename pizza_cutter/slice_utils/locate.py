import numpy as np


def build_slice_locations(*, central_size, buffer_size, image_width):
    """Build the locations of the slices.

    Parameters
    ----------
    central_size : int
        Size of the central region for metadetection in pixels.
    buffer_size : int
        Size of the buffer around the central region in pixels.
    image_width : int
        The width of the image in pixels.

    Returns
    -------
    row : array-like
        The row of the slice centers.
    col : array-like
        The column of the slice centers.
    start_row : array-like
        The starting row of the slice.
    start_col : array-like
        The starting column of the slice.
    """
    # compute the number of pizza slices
    # - each slice is central_size + 2 * buffer_size wide
    #   because a buffer is on each side
    # - each pizza slice is moved over by central_size to tile the full
    #  coadd tile
    # - we leave a buffer on either side so that we have to tile only
    #  im_width - 2 * buffer_size pixels
    cen_offset = (central_size - 1) / 2.0  # zero-indexed at pixel centers

    nobj = (image_width - 2 * buffer_size) / central_size
    if int(nobj) != nobj:
        raise ValueError(
            "The coadd tile must be exactly tiled by the pizza slices!")
    nobj = int(nobj)

    loc = np.arange(nobj) * central_size + buffer_size + cen_offset
    col, row = np.meshgrid(loc, loc)
    row = row.ravel()
    col = col.ravel()
    start_row = row - buffer_size - cen_offset
    start_col = col - buffer_size - cen_offset

    return row, col, start_row, start_col
