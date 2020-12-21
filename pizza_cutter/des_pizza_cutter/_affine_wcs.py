

class AffineWCS(object):
    """An affine WCS class which uses the esutil.wcsutil API.

    This class defines a mapping from image coordinates (x, y) to
    coordinates (u, v) that is defined by

        u = dudx * (x - x0) + dudy * (y - y0)
        v = dvdx * (x - x0) + dvdy * (y - y0)

    Note that (u,v) are traditionally in units of arcsec.

    Parameters
    ----------
    dudx : float
        The derivative of u wrt x for the Jacobian.
    dudy : float
        The derivative of u wrt y for the Jacobian.
    dvdx : float
        The derivative of v wrt x for the Jacobian.
    dvdy : float
        The derivative of v wrt y for the Jacobian.
    x0 : float
        The x/column part of the location of (u, v) = (0, 0) in pixel
        coordinates.
    y0 : float
        The y/row part of the location of (u, v) = (0, 0) in pixel
        coordinates.

    Methods
    -------
    image2sky(x, y)
        Convert image coordinates to sky coordinates.
    sky2image(u, v)
        Convert sky coordinates to image coordinates.
    get_jacobian(x, y)
        Get the local Jacobian of the affine transformation.
    is_celestial()
        Always returns False.

    Attributes
    ----------
    Any input parameter is also an attribute.
    """
    def __init__(self, *, dudx, dudy, dvdx, dvdy, x0, y0):
        self.dudx = dudx
        self.dudy = dudy
        self.dvdx = dvdx
        self.dvdy = dvdy
        self.x0 = x0
        self.y0 = y0

        self._det = dudx * dvdy - dvdx * dudy
        if self._det == 0:
            raise ValueError("Affine transformation is not invertible!")

    def __repr__(self):
        return "AffineWCS(dudx=%s, dudy=%s, dvdx=%s, dvdy=%s, x0=%s, y0=%s)" % (
            self.dudx,
            self.dudy,
            self.dvdx,
            self.dvdy,
            self.x0,
            self.y0,
        )

    def __str__(self):
        return self.__repr__()

    # you have to set both the __hash__ and __eq__ for the python functools.lru_cache
    # to work properly with this object
    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def image2sky(self, x, y):
        """Convert image coordinates to sky coordinates.

        Parameters
        ----------
        x : float or array-like
            The x/column coordinate usually in pixels.
        y : float or array-like
            The y/row coordinate in usually pixels.

        Returns
        -------
        u : float or array-like
            The u-coordinate usually in arcsec.
        v : float or array-like
            The v-coordinate usually in arcsec.
        """
        u = self.dudx * (x - self.x0) + self.dudy * (y - self.y0)
        v = self.dvdx * (x - self.x0) + self.dvdy * (y - self.y0)
        return u, v

    def sky2image(self, u, v):
        """Convert sky coordinates to image coordinates.

        Parameters
        -------
        u : float or array-like
            The u-coordinate usually in arcsec.
        v : float or array-like
            The v-coordinate usually in arcsec.

        Returns
        ----------
        x : float or array-like
            The x/column coordinate usually in pixels.
        y : float or array-like
            The y/row coordinate in usually pixels.
        """
        x = (self.dvdy * u - self.dudy * v) / self._det + self.x0
        y = (-self.dvdx * u + self.dudx * v) / self._det + self.y0
        return x, y

    def get_jacobian(self, x, y):
        """Get the local Jacobian of the affine transformation.

        Parameters
        ----------
        x : float or array-like
            Ignored. Present for API compatibility.
        y : float or array-like
            Ignored. Present for API compatibility.

        Returns
        -------
        jac : tuple of floats
            A tuple of the values (dudx, dudy, dvdx, dvdy).
        """
        return (
            self.dudx,
            self.dudy,
            self.dvdx,
            self.dvdy
        )

    def is_celestial(self):
        """Always returns False since this WCS is not defined on the sphere."""
        return False
