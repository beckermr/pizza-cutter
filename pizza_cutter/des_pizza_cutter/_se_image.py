import functools
import fitsio

import numpy as np
import esutil as eu
import galsim
import piff

from meds.bounds import Bounds
from meds.util import radec_to_uv
# import copy
#
from ..memmappednoise import MemMappedNoiseImage
from ._sky_bounds import get_rough_sky_bounds
from ._constants import MAGZP_REF


@functools.lru_cache(maxsize=16)
def _read_image(path, ext):
    """Cached reads of images.

    Each SE image in DES is ~33 MB in float. Thus we use at most ~0.5 GB of
    memory with a 16 element cache.
    """
    return fitsio.read(path, ext=ext)


class SEImageSlice(object):
    """A single-epoch image w/ associated metadata from the DES.

    NOTE: All pixel coordinates are with respect to the original image,
    not the coordinates in the slice. By convetion all pixel coordinates
    for this class are pixel-centered and **zero-indexed**. This convetion
    does not match the usual one-indexed coordinate convetion in galsim
    and the DES data products. Appropriate conversions are made as needed.

    Parameters
    ----------
    source_info : dict
        The single-epoch source info from the `desmeds` classes.
    psf_model : a `galsim.GSObject`, `galsim.des.DES_PSFEx`,
            or `piff.PSF` object
        The PSF model to use. The type of input will be detected and then
        called appropriately.
    wcs : a ` esutil.wcsutil.WCS` or `galsim.BaseWCS` instance
        The WCS model to use.
    noise_seed : int
        A seed to use for the noise field.

    Methods
    -------
    set_slice(x_start, y_start, box_size)
        Set the slice location and read a square slice of the
        image, weight map, bit mask, and the noise field.
    image2sky(x, y)
        Compute ra, dec for a given set of pixel coordinates.
    sky2image(ra, dec)
        Return x, y pixel coordinates for a given ra, dec.
    get_wcs_jacobian(x, y)
        Return the Jacobian of the WCS transformation as a `galsim.JacobianWCS`
        object.
    contains_radec(ra, dec)
        Use the approximate sky bounds to detect if the given point in
        ra, dec is anywhere in the total SE image (not just the slice).
    get_psf_image(x, y)
        Get an image of the PSF as a numpy array at the given location.

    Attributes
    ----------
    source_info : dict
        The source info dictionary for the SE image.
    image : np.ndarray
        The slice of the image.
    weight : np.ndarray
        The slice of the wight map.
    bmask : np.ndarray
        The slice of the bit mask.
    noise : np.ndarray
        The slice of the noise field for the image.
    x_start : int
        The zero-indexed starting column for the slice.
    y_start : int
        The zero-indexed starting row for the slice.
    box_size : int
        The size of the slice.
    ccd_bnds : meds.bounds.Bounds
        A boundary object for the full CCD.
    """
    def __init__(self, *, source_info, psf_model, wcs, noise_seed):
        self.source_info = source_info
        self._psf_model = psf_model
        self._wcs = wcs
        self._noise_seed = noise_seed

        # init the sky bounds
        sky_bnds, ra_ccd, dec_ccd = get_rough_sky_bounds(
            im_shape=(4096, 2048),
            wcs=self,  # the magic of APIs and duck typing - quack!
            position_offset=0,  # the wcs on this class is zero-indexed
            bounds_buffer_uv=16.0,  # in arcsec
            n_grid=4)
        self._sky_bnds = sky_bnds
        self._ra_ccd = ra_ccd
        self._dec_ccd = dec_ccd

        # init ccd bounds
        self.ccd_bnds = Bounds(0, 4096-1, 0, 2048-1)

    def set_slice(self, x_start, y_start, box_size):
        """Set the slice location and read a square slice of the
        image, weight map, bit mask, and the noise field.

        Parameters
        ----------
        x_start : int
            The zero-indexed x/column location for the start of the slice.
        y_start : int
            The zero-indexed y/row location for the start of the slice.
        box_size : int
            The size of the square slice.
        """
        self.x_start = x_start
        self.y_start = y_start
        self.box_size = box_size

        # we scale the image and weight map for the zero point
        scale = 10.0**(0.4*(MAGZP_REF - self.source_info['magzp']))

        # NOTE: DO NOT MODIFY THE UNDERLYING IMAGES IN PLACE BECAUSE THEY
        # ARE BEING CACHED!
        im = (
            _read_image(
                self.source_info['image_path'],
                ext=self.source_info['image_ext']) -
            _read_image(
                self.source_info['bkg_path'],
                ext=self.source_info['bkg_ext']))
        self.image = im[
            y_start:y_start+box_size, x_start:x_start+box_size].copy() * scale

        wgt = _read_image(
            self.source_info['weight_path'],
            ext=self.source_info['weight_ext'])
        self.weight = (
            wgt[y_start:y_start+box_size, x_start:x_start+box_size].copy() /
            scale**2)

        bmask = _read_image(
            self.source_info['bmask_path'],
            ext=self.source_info['bmask_ext'])
        self.bmask = bmask[
            y_start:y_start+box_size, x_start:x_start+box_size].copy()

        if not hasattr(self, '_noise'):
            self._noise = MemMappedNoiseImage(
                seed=self._noise_seed,
                weight=wgt / scale**2,
                sx=1024,
                sy=1024)

        self.noise = self._noise[
            y_start:y_start+box_size, x_start:x_start+box_size].copy()

    def image2sky(self, x, y):
        """Compute ra, dec for a given set of pixel coordinates.

        Parameters
        ----------
        x : scalar or 1d array-like
            The x/column image location in zero-indexed, pixel centered
            coordinates.
        y : scalar or 1d array-like
            The y/row image location in zero-indexed, pixel centered
            coordinates.

        Returns
        -------
        ra : scalar or 1d array-like
            The right ascension of the sky position.
        dec : scalar or 1d array-like
            The declination of the sky position.
        """
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            is_scalar = True
        else:
            is_scalar = False

        assert np.ndim(x) <= 1 and np.ndim(y) <= 1, (
            "Inputs to image2sky must be scalars or 1d arrays")
        assert np.ndim(x) == np.ndim(y), (
            "Inputs to image2sky must be the same shape")

        x = np.atleast_1d(x).ravel()
        y = np.atleast_1d(y).ravel()

        if isinstance(self._wcs, eu.wcsutil.WCS):
            # for the DES we always have a one-indexed system
            ra, dec = self._wcs.image2sky(x+1, y+1)
        elif isinstance(self._wcs, galsim.BaseWCS):
            assert self._wcs.isCelestial()
            ra = []
            dec = []
            for _x, _y in zip(x, y):
                # for the DES we always have a one-indexed system
                image_pos = galsim.PositionD(x=_x+1, y=_y+1)
                world_pos = self._wcs.toWorld(image_pos)
                _ra = world_pos.ra / galsim.degrees
                _dec = world_pos.dec / galsim.degrees
                ra.append(_ra)
                dec.append(_dec)
            ra = np.array(ra)
            dec = np.array(dec)
        else:
            raise ValueError('WCS %s not recognized!' % self._wcs)

        if is_scalar:
            return ra[0], dec[0]
        else:
            return ra, dec

    def sky2image(self, ra, dec):
        """Return x, y image coordinates for a given ra, dec.

        Parameters
        ----------
        ra : scalar or 1d array-like
            The right ascension of the sky position.
        dec : scalar or 1d array-like
            The declination of the sky position.

        Returns
        -------
        x : scalar or 1d array-like
            The x/column image location in zero-indexed, pixel centered
            coordinates.
        y : scalar or 1d array-like
            The y/row image location in zero-indexed, pixel centered
            coordinates.
        """
        if np.ndim(ra) == 0 and np.ndim(dec) == 0:
            is_scalar = True
        else:
            is_scalar = False

        assert np.ndim(ra) <= 1 and np.ndim(dec) <= 1, (
            "Inputs to sky2mage must be scalars or 1d arrays")
        assert np.ndim(ra) == np.ndim(dec), (
            "Inputs to sky2image must be the same shape")

        ra = np.atleast_1d(ra).ravel()
        dec = np.atleast_1d(dec).ravel()

        if isinstance(self._wcs, eu.wcsutil.WCS):
            # for the DES we always have a one-indexed system
            # so we subtract to get back to zero
            x, y = self._wcs.sky2image(ra, dec)
            x -= 1
            y -= 1
        elif isinstance(self._wcs, galsim.BaseWCS):
            assert self._wcs.isCelestial()

            x = []
            y = []
            for _ra, _dec in zip(ra, dec):
                # for the DES we always have a one-indexed system
                world_pos = galsim.CelestialCoord(
                    ra=_ra*galsim.degrees,
                    dec=_dec*galsim.degrees
                )
                image_pos = self._wcs.toImage(world_pos)
                x.append(image_pos.x - 1)
                y.append(image_pos.y - 1)
            x = np.array(x)
            y = np.array(y)
        else:
            raise ValueError('WCS %s not recognized!' % self._wcs)

        if is_scalar:
            return x[0], y[0]
        else:
            return x, y

    def get_wcs_jacobian(self, x, y):
        """Return the Jacobian of the WCS transformation as a
        `galsim.JacobianWCS` object.

        Parameters
        ----------
        x : scalar or 1d array-like
            The x/column image location in zero-indexed, pixel centered
            coordinates.
        y : scalar or 1d array-like
            The y/row image location in zero-indexed, pixel centered
            coordinates.

        Returns
        -------
        jac : galsim.JacobianWCS
            The WCS Jacobian at (x, y).
        """
        assert np.ndim(x) == 0 and np.ndim(y) == 0, (
            "WCS Jacobians are only returned for a single position at a time")

        if isinstance(self._wcs, eu.wcsutil.WCS):
            tup = self._wcs.get_jacobian(x+1, y+1)
            dudx = tup[0]
            dudy = tup[1]
            dvdx = tup[2]
            dvdy = tup[3]
            jac = galsim.JacobianWCS(
                dudx, dudy, dvdx, dvdy)
        elif isinstance(self._wcs, galsim.BaseWCS):
            pos = galsim.PositionD(x=x+1, y=y+1)
            jac = self._wcs.local(image_pos=pos)
        else:
            raise ValueError('WCS %s not recognized!' % self._wcs)

        return jac

    def contains_radec(self, ra, dec):
        """Use the approximate sky bounds to detect if the given point in
        ra, dec is anywhere in the total SE image (not just the slice).

        Parameters
        ----------
        ra : scalar or 1d array-like
            The right ascension of the sky position.
        dec : scalar or 1d array-like
            The declination of the sky position.

        Returns
        -------
        in_sky_bnds : bool or boolean array mask
            True if the point is in the approximate sky bounds, False
            otherwise.
        """
        if np.ndim(ra) == 0 and np.ndim(dec) == 0:
            is_scalar = True
        else:
            is_scalar = False

        assert np.ndim(ra) <= 1 and np.ndim(dec) <= 1, (
            "Inputs to sky2mage must be scalars or 1d arrays")
        assert np.ndim(ra) == np.ndim(dec), (
            "Inputs to sky2image must be the same shape")

        ra = np.atleast_1d(ra).ravel()
        dec = np.atleast_1d(dec).ravel()

        u, v = radec_to_uv(ra, dec, self._ra_ccd, self._dec_ccd)
        in_sky_bnds = self._sky_bnds.contains_points(u, v)

        if is_scalar:
            return bool(in_sky_bnds[0])
        else:
            return in_sky_bnds

    def get_psf_image(self, x, y):
        """Get an image of the PSF as a numpy array at the given location.

        Parameters
        ----------
        x : scalar
            The x/column image location in zero-indexed, pixel centered
            coordinates.
        y : scalar
            The y/row image location in zero-indexed, pixel centered
            coordinates.

        Returns
        -------
        psf_im : np.ndarray
            An image of the PSF with odd dimension and with the PSF centered
            at the canonical center of the image.
        """
        assert np.ndim(x) == 0 and np.ndim(y) == 0, (
            "PSFs are only returned for a single position at a time")

        # these are the subpixel offsets of the request position and
        # the nearest pixel center
        # we will draw the psf with these same offsets
        # when used with a coadd via interpolation, this should
        # locate the PSF center at the proper pixel location in the final
        # coadd
        dx = x - int(np.floor(x))
        dy = y - int(np.floor(y))

        if isinstance(self._psf_model, galsim.GSObject):

            # get jacobian
            wcs = self.get_wcs_jacobian(x, y)

            # draw the image
            if not hasattr(self, '_galsim_psf_dim'):
                im = self._psf_model.drawImage(
                    wcs=wcs, setup_only=True)
                if im.array.shape[0] % 2 == 0:
                    self._galsim_psf_dim = im.array.shape[0] + 1
                else:
                    self._galsim_psf_dim = im.array.shape[0]

            im = self._psf_model.drawImage(
                nx=self._galsim_psf_dim,
                ny=self._galsim_psf_dim,
                wcs=wcs,
                offset=galsim.PositionD(x=dx, y=dy))
            psf_im = im.array.copy()

        elif isinstance(self._psf_model, galsim.des.DES_PSFEx):
            # get the gs object
            psf = self._psf_model.getPSF(galsim.PositionD(x=x+1, y=y+1))

            # get the jacobian if a wcs is present
            if self._psf_model.wcs is None:
                # psf is in image coords, so jacobian is unity
                wcs = galsim.PixelScale(1.0)
            else:
                # psf is in work coords, so we need to draw with the jacobian
                wcs = self.get_wcs_jacobian(x, y)

            # draw the image
            im = psf.drawImage(
                nx=33, ny=33, wcs=wcs, method='no_pixel',
                offset=galsim.PositionD(x=dx, y=dy))
            psf_im = im.array.copy()

        elif isinstance(self._psf_model, piff.PSF):
            # draw the image
            # piff is zero offset? wtf?
            im = self._psf_model.draw(x=x, y=y, stamp_size=17)
            psf_im = im.array.copy()

        else:
            raise ValueError('PSF %s not recognized!' % self._psf_model)

        # always normalize to unity
        psf_im /= np.sum(psf_im)

        assert psf_im.shape[0] == psf_im.shape[1], "PSF image is not square!"
        assert psf_im.shape[0] % 2 == 1, "PSF dimension is not odd!"

        return psf_im
