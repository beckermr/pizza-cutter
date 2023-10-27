import esutil as eu
import numpy as np
import math

D2R = math.pi/180.0


def wrap_ra_diff(dra):
    """Given an input ra difference, wrap it to range -180, 180.

    Parameters
    ----------
    dra : float or np.ndarray
        The input difference in degrees.

    Returns
    -------
    wrapped_dra : float or np.ndarray
        Thje wrapped difference in degrees in the range [-180, 180].
    """
    if np.ndim(dra) == 0:
        if not np.isfinite(dra):
            return dra

        while dra < -180.0:
            dra += 360.0
        while dra > 180.0:
            dra -= 360.0
    else:
        msk_finite = np.isfinite(dra)
        msk = (dra < -180.0) & msk_finite
        while np.any(msk):
            dra[msk] = dra[msk] + 360.0
            msk = (dra < -180.0) & msk_finite

        msk = (dra > 180.0) & msk_finite
        while np.any(msk):
            dra[msk] = dra[msk] - 360.0
            msk = (dra > 180.0) & msk_finite

    return dra


class FastHashingWCS(eu.wcsutil.WCS):
    """It turns out hashing this WCS class is really slow and done a lot.

    This version has a stable hash/repr computed once.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = super().__repr__()

    def __repr__(self):
        return self._state

    def __str__(self):
        return self._state

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return hash(self) == hash(other)

    # we are overriding this one due to a bug - will push upstream
    def get_jacobian(self, x, y, distort=True, step=1.0):
        """
        Get the elementes of the jacobian matrix at the specified locations
        This method currently assumes the system is ra,dec

        parameters
        ----------
        x,y: scalars or arrays
            x and y coords in the image
        distort:  bool, optional
            Use the distortion model if present.  Default is True
        step: float
            Step used for central difference formula, in pixels.  Default is
            1.0 pixels.

        returns
        -------
        jacobian elements: tuple of arrays
            dra_dx, dra_dy, ddec_dx, ddec_dy

        method
        ------
        Finite difference
        """

        fac = 1.0/(2*step)

        ra, dec = self.image2sky(x, y, distort=distort)

        xp = x + step
        xm = x - step
        yp = y + step
        ym = y - step

        ra_p0, dec_p0 = self.image2sky(xp, y, distort=distort)
        ra_m0, dec_m0 = self.image2sky(xm, y, distort=distort)

        ra_0p, dec_0p = self.image2sky(x, yp, distort=distort)
        ra_0m, dec_0m = self.image2sky(x, ym, distort=distort)

        # in arcsec/pixel
        dra_dx = fac*3600.0*wrap_ra_diff(ra_p0-ra_m0)
        dra_dy = fac*3600.0*wrap_ra_diff(ra_0p-ra_0m)
        ddec_dx = fac*3600.0*(dec_p0-dec_m0)
        ddec_dy = fac*3600.0*(dec_0p-dec_0m)

        # need to scale dra b -cos(dec), minus sign since ra increases
        # to the left
        cosdec = -np.cos(dec*D2R)
        dra_dx *= cosdec
        dra_dy *= cosdec

        return dra_dx, dra_dy, ddec_dx, ddec_dy

    def sky2image(
        self, longitude, latitude, distort=True, find=False, xtol=eu.wcsutil.DEFTOL,
    ):
        """Invert the WCS. This method is a copy of the original but uses the inverse
        polynomial by default instead of root finding.

        Usage:
            x,y=sky2image(longitude, latitude, distort=True, find=True)

        Purpose:
            Convert between sky (lon,lat) and image coordinates (x,y)

        Inputs:
            longitude,latitude:  Probably ra,dec. Can be arrays.
        Optional Inputs:
            distort:  Use the distortion model if present.  Default is True
            find: When the distortion model is present, simply find the
                roots of the polynomial rather than using an inverse
                polynomial.  This is more accurate but slower. Default False.
            xtol: tolerance to use when root finding with find=True Default is
                1e-8.
        Outputs:
            x,y: x and y coords in the image.  Will have the same shape as
                lon,lat
        Example:
            from esutil import wcsutil
            import pyfits
            hdr=pyfits.getheader(fname)
            wcs = wcsutil.WCS(hdr)
            x,y = wcs.image2sky(ra,dec)
        """
        return super().sky2image(
            longitude, latitude, distort=distort, find=find, xtol=xtol
        )
