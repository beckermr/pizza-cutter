import numpy as np
import scipy.special


def measure_fwhm(image, smooth=0.1):
    """Measure the image FWHM.

    Parameters
    ----------
    image : 2-d array
        The image to measure.
    smooth : float
        The smoothing scale for the erf. This should be between 0 and 1. If
        you have noisy data, you might set this to the noise value or greater,
        scaled by the max value in the images.  Otherwise just make sure it
        smooths enough to avoid pixelization effects.

    Returns
    -------
    fwhm : float
        The FWHM in pixels.
    """

    thresh = 0.5
    nim = image.copy()
    maxval = image.max()
    nim /= maxval

    arg = (nim - thresh)/smooth

    vals = 0.5 * (1 + scipy.special.erf(arg))

    area = np.sum(vals)
    width = 2 * np.sqrt(area / np.pi)

    return width
