import functools
import logging
import time

import fitsio
import numpy as np
import esutil as eu
import galsim
import piff

from meds.bounds import Bounds
from meds.util import radec_to_uv

from ..coadding import (
    WCSInversionInterpolator,
    lanczos_resample,
    lanczos_resample_two,
)
from ..memmappednoise import MemMappedNoiseImage
from ._sky_bounds import get_rough_sky_bounds
from ._constants import MAGZP_REF, BMASK_EDGE

from ._tape_bumps import TAPE_BUMPS

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16)
def _read_image(path, ext):
    """Cached reads of images.

    Each SE image in DES is ~33 MB in float. Thus we use at most ~0.5 GB of
    memory with a 16 element cache.
    """
    return fitsio.read(path, ext=ext)


@functools.lru_cache(maxsize=16)
def _get_noise_image(weight_path, weight_ext, scale, noise_seed):
    """Cached generation of memory mapped noise images."""
    wgt = _read_image(weight_path, ext=weight_ext)
    zwgt_msk = wgt <= 0.0
    max_wgt = np.max(wgt[~zwgt_msk])

    return MemMappedNoiseImage(
        seed=noise_seed,
        weight=(wgt * (~zwgt_msk) + zwgt_msk * max_wgt) / scale**2,
        sx=1024, sy=1024)


@functools.lru_cache(maxsize=8)
def _get_wcs_inverse(wcs, wcs_position_offset, se_wcs, se_wcs_position_offset):

    if isinstance(se_wcs, galsim.BaseWCS):
        def _image2sky(x, y):
            try:
                ra, dec = se_wcs._radec(
                    x - se_wcs.x0 + se_wcs_position_offset,
                    y - se_wcs.y0 + se_wcs_position_offset)
                np.degrees(ra, out=ra)
                np.degrees(dec, out=dec)
                return ra, dec
            except AttributeError:
                return se_wcs.image2sky(
                    x+se_wcs_position_offset,
                    y+se_wcs_position_offset)

        dim_y = 4096
        dim_x = 2048
        delta = 8
        y_se, x_se = np.mgrid[:dim_y+delta:delta, :dim_x+delta:delta]
        y_se = y_se.ravel() - 0.5
        x_se = x_se.ravel() - 0.5

        x_coadd, y_coadd = wcs.sky2image(*_image2sky(x_se, y_se))

        x_coadd -= wcs_position_offset
        y_coadd -= wcs_position_offset

        return WCSInversionInterpolator(x_coadd, y_coadd, x_se, y_se)

    elif isinstance(se_wcs, eu.wcsutil.WCS):
        # return a closure here
        def _inv(x_coadd, y_coadd):
            x, y = se_wcs.sky2image(
                *wcs.image2sky(
                    x_coadd+wcs_position_offset,
                    y_coadd+wcs_position_offset),
                find=False)  # set find = False to make it fast!
            return x - se_wcs_position_offset, y - se_wcs_position_offset

        return _inv
    else:
        raise ValueError('WCS %s not recognized!' % se_wcs)


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
        image, weight map, bit mask, and the noise field. Sets the `image`
        and related attributes. See the attribute list below.
    image2sky(x, y)
        Compute ra, dec for a given set of pixel coordinates.
    sky2image(ra, dec)
        Return x, y pixel coordinates for a given ra, dec.
    get_wcs_jacobian(x, y)
        Return the Jacobian of the WCS transformation as a `galsim.JacobianWCS`
        object.
    ccd_contains_radec(ra, dec)
        Use the approximate sky bounds to detect if the given point in
        ra, dec is anywhere in the total SE image (not just the slice).
    ccd_contains_bounds(bounds)
        Uses the CCD bounds in image coordinates to test for an intersection
        with another bounds object.
    get_psf_image(x, y)
        Get an image of the PSF as a numpy array at the given location.
    compute_slice_bounds(ra, dec, box_size)
        Compute the patch bounds for a given ra,dec image and box size.
    set_psf(ra, dec)
        Set the PSF of the slice using the input (ra, dec). Sets the `psf`,
        `psf_x_start`, `psf_y_start` and `psf_box_size` attributes.
    resample(wcs, wcs_position_offset, x_start, y_start, box_size,
             psf_x_start, psf_y_start, psf_box_size)
        Resample a SEImageSlice to a new WCS.

    Attributes
    ----------
    source_info : dict
        The source info dictionary for the SE image.
    image : np.ndarray
        The slice of the image. Set by calling `set_slice`.
    weight : np.ndarray
        The slice of the wight map. Set by calling `set_slice`.
    bmask : np.ndarray
        The slice of the bit mask. Set by calling `set_slice`.
    noise : np.ndarray
        The slice of the noise field for the image. Set by calling `set_slice`.
    x_start : int
        The zero-indexed starting column for the slice. Set by calling
        `set_slice`.
    y_start : int
        The zero-indexed starting row for the slice. Set by calling
        `set_slice`.
    box_size : int
        The size of the slice. Set by calling `set_slice`.
    psf : np.ndarray
        An image of the PSF. Set by calling `set_psf`.
    psf_x_start : int
        The starting x/column location of the PSF image. Set by calling
        `set_psf`.
    psf_y_start : int
        The starting y/row location of the PSF image. Set by calling `set_psf`.
    psf_box_size : int
        The size of the PSF image. Set by calling `set_psf`.
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
        self._ccd_bnds = Bounds(0, 4096-1, 0, 2048-1)

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

        self._set_tape_bump_mask(bmask)

        self.bmask = bmask[
            y_start:y_start+box_size, x_start:x_start+box_size].copy()

        # this call rereads the weight image using the cached function
        # so it does not do any extra i/o
        # we are using the weight path and extension as part of the cache key
        nse = _get_noise_image(
            self.source_info['weight_path'], self.source_info['weight_ext'],
            scale, self._noise_seed)
        self.noise = nse[
            y_start:y_start+box_size, x_start:x_start+box_size].copy()

    def _set_tape_bump_mask(self, bmask):
        """
        set the tape bump flag on the input bmask

        Parameters
        ----------
        bmask: array
            This must be the original array before trimming

        Effects
        ------------
        The TAPEBUMP bit is set
        """

        ccdnum = self.source_info['ccdnum']
        bumps = TAPE_BUMPS[ccdnum]
        for bump in bumps:
            bmask[
                bump['row1']:bump['row2']+1,
                bump['col1']:bump['col2']+1,
            ] |= bump['flag']

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

            # ignoring color for now
            ra, dec = self._wcs._radec(
                x - self._wcs.x0 + 1, y - self._wcs.y0 + 1)
            np.degrees(ra, out=ra)
            np.degrees(dec, out=dec)
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

    def ccd_contains_radec(self, ra, dec):
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
            at the canonical center of the image. The central pixel of the
            returned image is located at coordinates `(int(x+0.5), int(y+0.5))`
            in the SE image.
        """
        assert np.ndim(x) == 0 and np.ndim(y) == 0, (
            "PSFs are only returned for a single position at a time")

        # - these are the subpixel offsets between the request position and
        #   the nearest pixel center
        # - we will draw the psf with these same offsets
        # - when used with a coadd via interpolation, this should
        #   locate the PSF center at the proper pixel location in the final
        #   coadd
        dx = x - int(x+0.5)
        dy = y - int(y+0.5)

        if isinstance(self._psf_model, galsim.GSObject):

            # get jacobian
            wcs = self.get_wcs_jacobian(x, y)

            # draw the image - here we cache galsim's intrnal image size
            # to force it to be off each time
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

            # draw the image - always use 33 pixels for DES Y3+
            im = psf.drawImage(
                nx=33, ny=33, wcs=wcs, method='no_pixel',
                offset=galsim.PositionD(x=dx, y=dy))
            psf_im = im.array.copy()

        elif isinstance(self._psf_model, piff.PSF):
            # draw the image
            # piff requires no offset since it renders in the actual
            # SE image pixel grid, not a hypothetical grid with the
            # star at a pixel center
            # again always 21 pixels for DES Y3+
            im = self._psf_model.draw(x=x+1, y=y+1, stamp_size=21)
            psf_im = im.array.copy()
        else:
            raise ValueError('PSF %s not recognized!' % self._psf_model)

        # always normalize to unity
        psf_im /= np.sum(psf_im)

        assert psf_im.shape[0] == psf_im.shape[1], "PSF image is not square!"
        assert psf_im.shape[0] % 2 == 1, "PSF dimension is not odd!"

        return psf_im

    def ccd_contains_bounds(self, bounds, buffer=0):
        """Uses the CCD bounds in image coordinates to test for an intersection
        with another bounds object.

        Parameters
        ----------
        bounds : meds.bounds.Bounds
            A bounds object to test with.
        buffer : int, optional
            An optional buffer amount by which to shrink the CCD boundry. This
            can be useful to exclude edge effects in the CCD.

        Returns
        -------
        intersects : bool
            True if the SE image intersects with this bounds object, False
            otherwise.
        """

        if buffer > 0:
            ccd_bnds = Bounds(
                self._ccd_bnds.rowmin + buffer,
                self._ccd_bnds.rowmax - buffer,
                self._ccd_bnds.colmin + buffer,
                self._ccd_bnds.colmax - buffer)
        else:
            ccd_bnds = self._ccd_bnds

        return ccd_bnds.contains_bounds(bounds)

    def compute_slice_bounds(self, ra, dec, box_size):
        """Compute the patch bounds for a given ra,dec image and box size.

        Parameters
        ----------
        ra : float
            The right ascension of the sky position.
        dec : float
            The declination of the sky position.
        box_size : int
            The size of the square slice.

        Returns
        -------
        patch_bounds : meds.bounds.Bounds
            The boundaries of the patch.
        """
        box_cen = (box_size - 1) / 2
        col, row = self.sky2image(ra, dec)
        se_start_row = int(row - box_cen + 0.5)
        se_start_col = int(col - box_cen + 0.5)
        patch_bnds = Bounds(
            rowmin=se_start_row,
            rowmax=se_start_row+box_size-1,
            colmin=se_start_col,
            colmax=se_start_col+box_size-1)

        return patch_bnds

    def set_psf(self, ra, dec):
        """Set the PSF of the slice using the input (ra, dec).

        Parameters
        ----------
        ra : float
            The right ascension of the sky position.
        dec : float
            The declination of the sky position.
        """

        x, y = self.sky2image(ra, dec)
        psf = self.get_psf_image(x, y)
        half = (psf.shape[0] - 1) / 2
        x_cen = int(x+0.5)
        y_cen = int(y+0.5)

        # make sure this is true so pixel index math is ok
        assert y_cen - half == int(y_cen - half)
        assert x_cen - half == int(x_cen - half)

        self.psf = psf
        self.psf_x_start = int(x_cen - half)
        self.psf_y_start = int(y_cen - half)
        self.psf_box_size = psf.shape[0]

    def resample(
            self, *, wcs, wcs_position_offset, x_start, y_start, box_size,
            psf_x_start, psf_y_start, psf_box_size):
        """Resample a SEImageSlice to a new WCS.

        The resampling is done as follows. For each pixel in the destination
        image, we compute its location in the source image. Then we use a
        Lanczos3 interpolant in the source grid to compute the value of the
        pixel in the destination image. For bit masks, we use the nearest pixel
        instead of an interpolant.

        NOTE: For destination grids that are far from the input grid, this
        function will not work.

        NOTE: Typically, the input WCS object will be for a coadd.

        Parameters
        ----------
        wcs : `esutil.wcsutil.WCS` object
            The WCS model to use for the resampled pixel grid. This object is
            assumed to take one-indexed, pixel-centered coordinates.
        wcs_position_offset : int
            The coordinate offset, if any, to get from the zero-indexed, pixel-
            centered coordinates used by this class to the coordinate convetion
            of the input `wcs` object.
        x_start : int
            The zero-indexed, pixel-centered starting x/column in the
            destination grid.
        y_start : int
            The zero-indexed, pixel-centered starting y/row in the
            destination grid.
        box_size : int
            The size of the square region to resample to in the destination
            grid.
        psf_x_start : int
            The zero-indexed, pixel-centered starting x/column in the
            destination PSF grid.
        psf_y_start : int
            The zero-indexed, pixel-centered starting y/row in the
            destination PSF grid.
        psf_box_size : int
            The size of the square region to resample to in the destination
            PSF grid.

        Returns
        -------
        resampled_data : dict
            A dictionary with the resampled data. It has keys

                'image' : the resampled image
                'bmask' : an approximate bmask using the nearest SE image
                    pixel
                'noise' : the resampled noise image
                'psf' : the resmapled PSF image
        """
        # error check
        if not hasattr(self, 'box_size'):
            raise RuntimeError("You must call set_slice before resmpling!")

        if not hasattr(self, 'psf_box_size'):
            raise RuntimeError("You must call set_psf before resmpling!")

        # 1. build the lookup table of SE position as a function of coadd
        # position
        # we always go SE -> sky -> coadd because inverting the SE WCS will
        # be more expensive since they typically have distortion/tree ring
        # terms which require a root finder
        # we use a buffer to make sure edge pixels are ok
        logger.debug('start wcs interp')
        t0 = time.time()
        wcs_interp = _get_wcs_inverse(
            wcs, wcs_position_offset,
            self._wcs, self.source_info['position_offset'])
        logger.debug('end wcs interp: %f', time.time() - t0)

        # 2. using the lookup table, we resample each image to the
        # coadd coordinates

        # compute the SE image positions using the lookup table
        y_rs, x_rs = np.mgrid[0:box_size, 0:box_size]
        y_rs = y_rs.ravel()
        x_rs = x_rs.ravel()
        # we need to add in the zero-indexed lower, left location of the
        # slice of the coadd image here since the WCS interp is in
        # absolute image pixel units
        x_rs_se, y_rs_se = wcs_interp(x_rs + x_start, y_rs + y_start)

        # remove the offset to local coords for resampling
        x_rs_se -= self.x_start
        y_rs_se -= self.y_start

        rim, rn = lanczos_resample_two(
            self.image,
            self.noise,
            y_rs_se,
            x_rs_se
        )
        resampled_data = {
            'image': rim.reshape(box_size, box_size),
            'noise': rn.reshape(box_size, box_size),
        }

        # 3. do the nearest pixel for the bit mask
        y_rs_se = (y_rs_se + 0.5).astype(np.int64)
        x_rs_se = (x_rs_se + 0.5).astype(np.int64)
        msk = (
            (y_rs_se >= 0) & (y_rs_se < self.bmask.shape[0]) &
            (x_rs_se >= 0) & (x_rs_se < self.bmask.shape[1]))
        bmask = np.zeros((box_size, box_size), dtype=self.bmask.dtype)
        bmask[y_rs[msk], x_rs[msk]] = self.bmask[y_rs_se[msk], x_rs_se[msk]]
        bmask[y_rs[~msk], x_rs[~msk]] = BMASK_EDGE
        resampled_data['bmask'] = bmask

        # 4. do the PSF image
        y_rs, x_rs = np.mgrid[0:psf_box_size, 0:psf_box_size]
        y_rs = y_rs.ravel() + psf_y_start
        x_rs = x_rs.ravel() + psf_x_start
        x_rs_se, y_rs_se = wcs_interp(x_rs, y_rs)

        # remove the offset to local coords for resampling
        x_rs_se -= self.psf_x_start
        y_rs_se -= self.psf_y_start

        resampled_data['psf'] = lanczos_resample(
            self.psf, y_rs_se, x_rs_se).reshape(psf_box_size, psf_box_size)

        return resampled_data
