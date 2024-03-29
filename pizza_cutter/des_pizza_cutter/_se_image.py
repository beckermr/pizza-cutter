import logging
import time
import pprint
from functools import lru_cache
import os
import copy

import fitsio
import numpy as np
import esutil as eu
import galsim
import piff
import pixmappy

from meds.bounds import Bounds
from meds.util import radec_to_uv

from ..coadding import (
    WCSInversionInterpolator,
    WCSGridScalarInterpolator,
    lanczos_resample,
    lanczos_resample_two,
)
from ..memmappednoise import MemMappedNoiseImage
from ._sky_bounds import get_rough_sky_bounds
from ._constants import (
    MAGZP_REF,
    BMASK_EDGE,
    BMASK_RESAMPLE_BOUNDS,
)
from ._affine_wcs import AffineWCS
from ._tape_bumps import TAPE_BUMPS
from ._load_info import _munge_fits_header
from ._piff_tools import get_piff_psf
from ..wcs import wrap_ra_diff, FastHashingWCS

logger = logging.getLogger(__name__)

SMALL_IMAGE_CACHE_SIZE = 32
BIG_IMAGE_CACHE_SIZE = 512

# TODO: make a config option?
PIFF_STAMP_SIZE = 25


@lru_cache(maxsize=BIG_IMAGE_CACHE_SIZE)
def _cached_get_rough_sky_bounds(
    *,
    im_shape, wcs, position_offset, bounds_buffer_uv, n_grid, celestial
):
    return get_rough_sky_bounds(
        im_shape=im_shape,
        wcs=wcs,  # the magic of APIs and duck typing - quack!
        position_offset=position_offset,
        bounds_buffer_uv=bounds_buffer_uv,  # in arcsec
        n_grid=n_grid,
        celestial=celestial,
    )


@lru_cache(maxsize=BIG_IMAGE_CACHE_SIZE)
def _get_image_shape(*, image_path, image_ext):
    h = fitsio.read_header(image_path, ext=image_ext)
    if 'znaxis1' in h:
        return h['znaxis2'], h['znaxis1']
    else:
        return h['naxis2'], h['naxis1']


def _read_image(path, ext):
    if isinstance(path, np.ndarray):
        return path
    else:
        return _read_image_cached(path, ext)


@lru_cache(maxsize=SMALL_IMAGE_CACHE_SIZE*2)
def _read_image_cached(path, ext):
    """Cached reads of images.

    Each SE image in DES is ~33 MB in float. Thus we use at most ~0.5 GB of
    memory with a 16 element cache.
    """
    ci = _read_image_cached.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_read_image_cached cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )
    return fitsio.read(path, ext=ext)


def _get_noise_image_impl(weight_path, weight_ext, scale, noise_seed, tmpdir):
    wgt = _read_image(weight_path, ext=weight_ext)
    zwgt_msk = wgt <= 0.0
    max_wgt = np.max(wgt[~zwgt_msk])

    return MemMappedNoiseImage(
        seed=noise_seed,
        weight=(wgt * (~zwgt_msk) + zwgt_msk * max_wgt) / scale**2,
        dir=tmpdir,
        sx=1024, sy=1024,
    )


def _get_noise_image(weight_path, weight_ext, scale, noise_seed, tmpdir):
    """Cached generation of memory mapped noise images."""
    if isinstance(weight_path, np.ndarray):
        return _get_noise_image_impl(weight_path, weight_ext, scale, noise_seed, tmpdir)
    else:
        return _get_noise_image_cached(
            weight_path, weight_ext, scale, noise_seed, tmpdir
        )


@lru_cache(maxsize=SMALL_IMAGE_CACHE_SIZE)
def _get_noise_image_cached(weight_path, weight_ext, scale, noise_seed, tmpdir):
    """Cached generation of memory mapped noise images."""
    ci = _get_noise_image_cached.cache_info()
    if ci.misses == ci.maxsize + 1:
        print(
            "_get_noise_image_cached cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    return _get_noise_image_impl(weight_path, weight_ext, scale, noise_seed, tmpdir)


@lru_cache(maxsize=SMALL_IMAGE_CACHE_SIZE)
def _get_wcs_inverse(wcs, wcs_position_offset, se_wcs, se_im_shape, delta):
    ci = _get_wcs_inverse.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_get_wcs_inverse cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    if hasattr(se_wcs, "source_info"):
        logger.debug(
            "wcs inverse cache miss for %s/%s",
            se_wcs.source_info["path"],
            se_wcs.source_info["filename"],
        )
    else:
        logger.debug("wcs inverse cache miss for %s", se_wcs)

    dim_y = se_im_shape[0]
    dim_x = se_im_shape[1]
    y_se, x_se = np.mgrid[:dim_y+delta:delta, :dim_x+delta:delta]
    y_se = y_se.ravel() - 0.5
    x_se = x_se.ravel() - 0.5

    x_coadd, y_coadd = wcs.sky2image(*se_wcs.image2sky(x_se, y_se))

    x_coadd -= wcs_position_offset
    y_coadd -= wcs_position_offset

    return (
        WCSInversionInterpolator(x_coadd, y_coadd, x_se, y_se),
        WCSInversionInterpolator(x_se, y_se, x_coadd, y_coadd),
    )


def _compute_wcs_area(se_wcs, x_se, y_se, dxy=1):
    ra, dec = se_wcs.image2sky(x_se, y_se)
    ra_xp, dec_xp = se_wcs.image2sky(x_se + dxy, y_se)
    ra_xm, dec_xm = se_wcs.image2sky(x_se - dxy, y_se)
    ra_yp, dec_yp = se_wcs.image2sky(x_se, y_se + dxy)
    ra_ym, dec_ym = se_wcs.image2sky(x_se, y_se - dxy)

    if isinstance(se_wcs, eu.wcsutil.WCS) or se_wcs.is_celestial():
        # code here follows the computation in galsim or esutil
        cosdec = np.cos(dec * (np.pi / 180.0))
        dudx = -0.5 * wrap_ra_diff(ra_xp - ra_xm) / dxy * cosdec * 3600
        dudy = -0.5 * wrap_ra_diff(ra_yp - ra_ym) / dxy * cosdec * 3600
        dvdx = 0.5 * (dec_xp - dec_xm) / dxy * 3600
        dvdy = 0.5 * (dec_yp - dec_ym) / dxy * 3600
    else:
        dudx = 0.5 * (ra_xp - ra_xm) / dxy
        dudy = 0.5 * (ra_yp - ra_ym) / dxy
        dvdx = 0.5 * (dec_xp - dec_xm) / dxy
        dvdy = 0.5 * (dec_yp - dec_ym) / dxy

    return np.abs(dudx * dvdy - dvdx * dudy)


@lru_cache(maxsize=SMALL_IMAGE_CACHE_SIZE)
def _get_wcs_area_interp(se_wcs, se_im_shape, delta, position_offset=0):
    ci = _get_wcs_area_interp.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_get_wcs_area_interp cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    if hasattr(se_wcs, "source_info"):
        logger.debug(
            "wcs area interp cache miss for %s/%s",
            se_wcs.source_info["path"],
            se_wcs.source_info["filename"],
        )
    else:
        logger.debug("wcs area interp cache miss for %s", se_wcs)

    dim_y = se_im_shape[0]
    dim_x = se_im_shape[1]
    y_se, x_se = np.mgrid[:dim_y+delta:delta, :dim_x+delta:delta]
    shape = y_se.shape
    y_se = y_se.ravel() - 0.5
    x_se = x_se.ravel() - 0.5

    area = _compute_wcs_area(se_wcs, x_se + position_offset, y_se + position_offset)
    area = area.reshape(shape).T  # put x dim first

    return WCSGridScalarInterpolator(
        np.mgrid[:dim_x+delta:delta],
        np.mgrid[:dim_y+delta:delta],
        area,
    )


@lru_cache(maxsize=BIG_IMAGE_CACHE_SIZE)
def _load_piff_pixmappy(piff_path, chipnum):
    ci = _load_piff_pixmappy.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_load_piff_pixmappy cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    logger.debug("load Piff miss for %s", piff_path)
    piff_path = os.path.expandvars(piff_path)
    psf = get_piff_psf(piff_path)

    # try and grab pixmappy from piff
    wcs = psf.wcs[chipnum]
    if isinstance(wcs, pixmappy.GalSimWCS):
        # HACK at the internals to code around a bug!
        if isinstance(
                wcs.origin,
                galsim._galsim.PositionD):
            logger.warning(
                "adjusting the pixmappy origin to fix a bug!"
            )
            wcs._origin = galsim.PositionD(
                wcs._origin.x,
                wcs._origin.y
            )
    else:
        raise RuntimeError(
            "Could not extract pixmappy WCS from piff model %s" % piff_path
        )
    return psf, wcs


@lru_cache(maxsize=BIG_IMAGE_CACHE_SIZE)
def _load_psfex(psfex_path):
    ci = _load_psfex.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_load_psfex cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    logger.debug("load psfex cache miss for %s", psfex_path)
    psfex_path = os.path.expandvars(psfex_path)
    return galsim.des.DES_PSFEx(psfex_path)


@lru_cache(maxsize=BIG_IMAGE_CACHE_SIZE)
def _load_image_wcs(image_path, image_ext):
    ci = _load_image_wcs.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_load_image_wcs cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    logger.debug("load wcs cache miss for %s[%s]", image_path, image_ext)
    return FastHashingWCS(
        _munge_fits_header(
            fitsio.read_header(
                os.path.expandvars(image_path), ext=image_ext
            )
        )
    )


@lru_cache(maxsize=BIG_IMAGE_CACHE_SIZE)
def _load_head_wcs(head_path, image_path, image_ext):
    ci = _load_head_wcs.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_load_head_wcs cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    logger.debug("load wcs cache miss for %s", head_path)

    im_hdr = _munge_fits_header(
        fitsio.read_header(
            os.path.expandvars(image_path), ext=image_ext,
        )
    )
    hdr = _munge_fits_header(
        fitsio.read_scamp_head(os.path.expandvars(head_path))
    )
    for key in ["naxis1", "naxis2", "znaxis1", "znaxis2"]:
        if key in im_hdr:
            hdr[key] = im_hdr[key]

    return FastHashingWCS(hdr)


@lru_cache(maxsize=BIG_IMAGE_CACHE_SIZE)
def _compute_bad_piff_model_delta_flag(
    *, piff_path, ccdnum, grid_size, seed, piff_tuples, max_abs_t_diff,
):
    ci = _compute_bad_piff_model_delta_flag.cache_info()
    if ci.misses == ci.maxsize+1:
        print(
            "_compute_bad_piff_model_delta_flag cache size exceeded: maxsize = %d" % (
                ci.maxsize,
            ),
            flush=True,
        )

    logger.debug("Piff bad mask cache miss for %s", piff_path)

    from des_y6utils.piff import measure_t_grid_for_piff_model

    piff_model = _load_piff_pixmappy(
        piff_path,
        ccdnum,
    )[0]

    piff_kwargs = {k: v for k, v in piff_tuples}

    res = measure_t_grid_for_piff_model(
        piff_model, piff_kwargs, seed=seed, grid_size=grid_size
    )
    has_nans = np.any(~np.isfinite(res))
    abs_t_diff = np.nanmax(np.abs(res - np.nanmedian(res)))

    if has_nans or abs_t_diff > max_abs_t_diff:
        return True
    else:
        return False


def _check_point_in_bad_piff_model_mask(x, y, bad_msk, grid_size):
    xg = np.clip(np.floor(x/grid_size), 0, bad_msk.shape[1]-1).astype(int)
    yg = np.clip(np.floor(y/grid_size), 0, bad_msk.shape[0]-1).astype(int)
    return bad_msk[yg, xg]


def clear_image_and_wcs_caches():
    """Clear the global image and WCS caches."""
    _get_image_shape.cache_clear()
    _read_image_cached.cache_clear()
    _get_noise_image_cached.cache_clear()
    _get_wcs_inverse.cache_clear()
    _get_wcs_area_interp.cache_clear()
    _cached_get_rough_sky_bounds.cache_clear()
    _load_psfex.cache_clear()
    _load_image_wcs.cache_clear()
    _load_head_wcs.cache_clear()
    _load_piff_pixmappy.cache_clear()
    _compute_bad_piff_model_delta_flag.cache_clear()


class SEImageSlice(object):
    """A single-epoch image w/ associated metadata.

    NOTE: All pixel coordinates are with respect to the original image,
    not the coordinates in the slice. By convetion all pixel coordinates
    for this class are pixel-centered and **zero-indexed**. This convetion
    does not match the usual one-indexed coordinate convetion in galsim
    and the DES data products. Appropriate conversions are made as needed.

    Parameters
    ----------
    source_info : dict
        The single-epoch source info. See the DES `desmeds` format for the
        possible keys.
    psf_model : a `galsim.GSObject`, `galsim.des.DES_PSFEx`,
            or `piff.PSF` object
        The PSF model to use. The type of input will be detected and then
        called appropriately.
    psf_kwargs : dict or None
        A set of keywords to feed the PSF model call.
    wcs : a ` esutil.wcsutil.WCS`, `AffineWCS` or `galsim.BaseWCS` instance
        The WCS model to use.
    wcs_position_offset : float
        The offset to get from pixel-centered, zero-indexed coordinates to
        the coordinates expected by the WCS.
    wcs_color : float
        A color to use for the WCS. Typically zero is fine, but for pixmappy
        it is worth thinking about this. A good default might be 0.61.
    noise_seeds : list of int
        The list of seeds to use for the noise fields.
    mask_tape_bumps: bool
        If True, turn on TAPEBUMP flag and turn off SUSPECT in bmask for
        tape bump regions in DES CCDs.
    mask_piff_failure_config : dict or None
        If not None, then a dictionary with the parameters used to mask regions of
        Piff models where the fit appears to fail. Required keys are:

            grid_size : int
                Set to something that divides the SE image evenly. 128 is a
                good choice.
            max_abs_T_diff : float
                The maximum allowed absolute difference from the median PSF model
                T allowed. Default of 0.15 is pretty good.
            seed : int
                An RNG seed to use.

        The algorithm computes the value of T on a grid on the CCD. If any T
        values come out as NaN or the max deviation from the median is greater
        than `max_abs_T_diff`, then the whole CCD is flagged as bad.
    tmpdir: optional, string
        Optional temporary directory for temporary files

    Methods
    -------
    set_slice(patch_bnds)
        Set the slice location and read a square slice of the
        image, weight map, bit mask, and the noise field. Sets the `image`
        and related attributes. See the attribute list below.
    image2sky(x, y)
        Compute ra, dec for a given set of pixel coordinates.
    sky2image(ra, dec)
        Return x, y pixel coordinates for a given ra, dec.
    get_wcs_pixel_area(x, y)
        Get the pixel scale at a set of x-y locations.
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
    compute_slice_bounds(ra, dec, box_size, frac_buffer)
        Compute the patch bounds for a given ra,dec image and box size.
    set_psf(ra, dec)
        Set the PSF of the slice using the input (ra, dec). Sets the `psf`,
        `psf_x_start`, `psf_y_start` and `psf_box_size` attributes.
    resample(
        wcs, wcs_position_offset, x_start, y_start, box_size,
        psf_x_start, psf_y_start, psf_box_size,
        se_wcs_interp_delta, coadd_wcs_interp_delta
    )
        Resample a SEImageSlice to a new WCS.
    set_interp_image_noise_pmask(interp_image, interp_noises, mask)
        Set the inteprolated image, noise fields, and processing mask.

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
    noises : list of np.ndarray
        The slices of the noise fields for the image. Set by calling `set_slice`.
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

    If `set_interp_image_noise_pmask` is called then the following attributes
    are set:

    pmask : np.ndarray
        The image of processing flags.
    interp_frac : np.ndarray
        The fraction of each pixel that is interpolated.
    orig_image : np.ndarray
        The original image without interpolation.
    interp_only_image : np.ndarray
        An image of only the interpolated flux.
    """
    def __init__(self,
                 *,
                 source_info,
                 psf_model,
                 psf_kwargs,
                 wcs,
                 wcs_position_offset,
                 wcs_color,
                 noise_seeds,
                 mask_tape_bumps,
                 mask_piff_failure_config,
                 tmpdir=None):

        self.source_info = source_info
        self._wcs_position_offset = wcs_position_offset
        self._wcs_color = wcs_color
        self._noise_seeds = noise_seeds
        self._mask_tape_bumps = mask_tape_bumps
        self._mask_piff_failure_config = mask_piff_failure_config
        self._tmpdir = tmpdir

        if isinstance(wcs, str):
            self._wcs_type = wcs

            if wcs == 'image':
                wcs = _load_image_wcs(
                    source_info['image_path'],
                    source_info['image_ext'],
                )
            elif wcs == "head":
                wcs = _load_head_wcs(
                    source_info["head_path"],
                    source_info['image_path'],
                    source_info['image_ext'],
                )
            elif wcs == 'affine':
                wcs = source_info['affine_wcs']
            elif wcs == 'pixmappy':
                res = _load_piff_pixmappy(
                    source_info['piff_path'],
                    source_info["ccdnum"],
                )
                wcs = res[1]
            else:
                raise RuntimeError("wcs type %s not allowed!" % wcs)
        else:
            self._wcs_type = None

        if isinstance(psf_model, str):
            if psf_model == 'galsim':
                psf_model = source_info['galsim_psf']
            elif psf_model == 'psfex':
                psf_model = _load_psfex(source_info['psfex_path'])
            elif psf_model == 'piff':
                res = _load_piff_pixmappy(
                    source_info['piff_path'],
                    source_info["ccdnum"],
                )
                psf_model = res[0]
            else:
                raise RuntimeError("psf type %s not allowed!" % psf_model)

        self._psf_model = psf_model
        self._psf_kwargs = psf_kwargs or {}
        self._wcs = wcs

        # get the image shape
        if 'image_shape' in source_info:
            self._im_shape = source_info['image_shape']
        else:
            self._im_shape = _get_image_shape(
                image_path=source_info['image_path'],
                image_ext=source_info['image_ext'],
            )

        # init the sky bounds
        if (isinstance(wcs, AffineWCS) or
                (isinstance(wcs, galsim.BaseWCS) and not wcs.isCelestial())):
            self._wcs_is_celestial = False
        else:
            self._wcs_is_celestial = True

        sky_bnds, ra_ccd, dec_ccd = _cached_get_rough_sky_bounds(
            im_shape=self._im_shape,
            wcs=self,  # the magic of APIs and duck typing - quack!
            position_offset=0,  # the wcs on this class is zero-indexed
            bounds_buffer_uv=16.0,  # in arcsec
            n_grid=4,
            celestial=self._wcs_is_celestial,
        )
        self._sky_bnds = sky_bnds
        self._ra_ccd = ra_ccd
        self._dec_ccd = dec_ccd

        # init ccd bounds
        self._ccd_bnds = Bounds(0, self._im_shape[0]-1, 0, self._im_shape[1]-1)

    def __repr__(self):
        if not hasattr(self, "_state"):
            state = {}
            state.update(self.source_info)
            state["__internals"] = {}
            for attr in [
                "_psf_model",
                "_wcs",
                "_wcs_position_offset",
                "_wcs_color",
                "_noise_seeds",
                "_mask_tape_bumps",
            ]:
                state["__internals"][attr] = repr(getattr(self, attr))
            self._state = "SEImageSlice: " + pprint.pformat(state)

        return self._state

    def __str__(self):
        return self.__repr__()

    # you have to set both the __hash__ and __eq__ for the python functools.lru_cache
    # to work properly with this object
    # for now the repr is not eval-able since its inputs are not.
    # thus I set it to something very unique and wrote a test that this is working
    # the image name and path is in the __repr__ along with the actual PSF object
    # objects

    def __hash__(self):
        if not hasattr(self, "_state_hash"):
            self._state_hash = hash(self.__repr__())
        return self._state_hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def is_celestial(self):
        if (
            isinstance(self._wcs, AffineWCS)
            or isinstance(self._wcs, galsim.wcs.EuclideanWCS)
        ):
            return False
        else:
            return True

    def set_slice(self, patch_bnds):
        """Set the slice location and read a square slice of the
        image, weight map, bit mask, and the noise field.

        Parameters
        ----------
        patch_bnds : meds.bounds.Bounds
            A Bounds object encoding the square slice bounds.
        """
        x_start = patch_bnds.colmin
        y_start = patch_bnds.rowmin
        box_size = patch_bnds.colmax - patch_bnds.colmin + 1
        assert box_size == patch_bnds.rowmax - patch_bnds.rowmin + 1

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

        bmask = _read_image(
            self.source_info['bmask_path'],
            ext=self.source_info['bmask_ext'])

        if self._mask_tape_bumps:
            self._set_tape_bump_mask(bmask)

        self.bmask = bmask[
            y_start:y_start+box_size, x_start:x_start+box_size].copy()

        wgt = _read_image(
            self.source_info['weight_path'],
            ext=self.source_info['weight_ext'])
        self.weight = (
            wgt[y_start:y_start+box_size, x_start:x_start+box_size].copy() /
            scale**2)

        # this call rereads the weight image using the cached function
        # so it does not do any extra i/o
        # we are using the weight path and extension as part of the cache key
        noises = []
        for noise_seed in self._noise_seeds:
            nse = _get_noise_image(
                self.source_info['weight_path'], self.source_info['weight_ext'],
                scale, noise_seed,
                self._tmpdir,
            )
            noises.append(nse[
                y_start:y_start+box_size, x_start:x_start+box_size].copy()
            )
        self.noises = noises

    def _set_tape_bump_mask(self, bmask):
        """
        set the TAPEBUMP flag on the input bmask, unset
        SUSPECT

        Parameters
        ----------
        bmask: array
            This must be the original array before trimming

        Effects
        ------------
        The TAPEBUMP bit is set and SUSPECT is unset
        """

        logger.info('masking tape bumps')

        ccdnum = self.source_info['ccdnum']
        bumps = TAPE_BUMPS[ccdnum]
        suspect = 2048
        for bump in bumps:
            bmask[
                bump['row1']:bump['row2']+1,
                bump['col1']:bump['col2']+1,
            ] |= bump['flag']
            bmask[
                bump['row1']:bump['row2']+1,
                bump['col1']:bump['col2']+1,
            ] &= ~suspect

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

        if (isinstance(self._wcs, eu.wcsutil.WCS) or
                isinstance(self._wcs, AffineWCS)):
            ra, dec = self._wcs.image2sky(
                x + self._wcs_position_offset,
                y + self._wcs_position_offset)
        elif isinstance(self._wcs, galsim.BaseWCS):
            assert self._wcs.isCelestial()
            ra, dec = self._wcs.xyToradec(
                x + self._wcs_position_offset,
                y + self._wcs_position_offset,
                units=galsim.degrees,
                color=self._wcs_color,
            )
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

        if (isinstance(self._wcs, eu.wcsutil.WCS) or
                isinstance(self._wcs, AffineWCS)):
            x, y = self._wcs.sky2image(ra, dec)
            x -= self._wcs_position_offset
            y -= self._wcs_position_offset
        elif isinstance(self._wcs, galsim.BaseWCS):
            assert self._wcs.isCelestial()

            x, y = self._wcs.radecToxy(
                ra,
                dec,
                galsim.degrees,
                color=self._wcs_color,
            )
            x -= self._wcs_position_offset
            y -= self._wcs_position_offset
        else:
            raise ValueError('WCS %s not recognized!' % self._wcs)

        if is_scalar:
            return x[0], y[0]
        else:
            return x, y

    def get_wcs_pixel_area(self, x, y):
        """Get the pixel scale at a set of x-y locations.

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
        pixel_area : scalar or 1d array-like
            The pixel area in arcsec**2.
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

        area = _compute_wcs_area(self, x, y)

        if is_scalar:
            return area[0]
        else:
            return area

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

        if (isinstance(self._wcs, eu.wcsutil.WCS) or
                isinstance(self._wcs, AffineWCS)):
            tup = self._wcs.get_jacobian(
                x + self._wcs_position_offset,
                y + self._wcs_position_offset)
            dudx = tup[0]
            dudy = tup[1]
            dvdx = tup[2]
            dvdy = tup[3]
            jac = galsim.JacobianWCS(
                dudx, dudy, dvdx, dvdy)
        elif isinstance(self._wcs, galsim.BaseWCS):
            pos = galsim.PositionD(
                x=x + self._wcs_position_offset,
                y=y + self._wcs_position_offset)
            jac = self._wcs.local(image_pos=pos, color=self._wcs_color)
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

        if self._wcs_is_celestial:
            u, v = radec_to_uv(ra, dec, self._ra_ccd, self._dec_ccd)
        else:
            u = ra - self._ra_ccd
            v = dec - self._dec_ccd

        in_sky_bnds = self._sky_bnds.contains_points(u, v)

        if is_scalar:
            return bool(in_sky_bnds[0])
        else:
            return in_sky_bnds

    def _compute_psf_stamp_bounds(self, x, y, dim):
        # compute the lower left corner of the stamp
        # we find the nearest pixel to the input (x, y)
        # and offset by half the stamp size in pixels
        # assumes the stamp size is odd
        # there is an assert for this below
        half = (dim - 1) / 2
        x_cen = np.floor(x+0.5)
        y_cen = np.floor(y+0.5)

        # make sure this is true so pixel index math is ok
        assert y_cen - half == int(y_cen - half)
        assert x_cen - half == int(x_cen - half)

        # compute bounds in Piff wcs coords
        xmin = int(x_cen - half)
        ymin = int(y_cen - half)

        return xmin, ymin

    def get_psf_image(self, x, y, psf_kwargs=None):
        """Get an image of the PSF as a numpy array at the given location.

        Parameters
        ----------
        x : scalar
            The x/column image location in zero-indexed, pixel centered
            coordinates.
        y : scalar
            The y/row image location in zero-indexed, pixel centered
            coordinates.
        psf_kwargs : None or dict, optional
            If not None, the PSF will be drawn with these kwargs passed to the Piff
            call when using Piff.

        Returns
        -------
        psf_im : np.ndarray
            An image of the PSF with odd dimension and with the PSF centered
            at the subpixel offset (x - floor(x+0.5), y - floor(y+0.5))
            relative to the true center of the image.
        """
        assert np.ndim(x) == 0 and np.ndim(y) == 0, (
            "PSFs are only returned for a single position at a time")

        # - these are the subpixel offsets between the request position and
        #   the nearest pixel center
        # - we will draw the psf with these same offsets
        # - when used with a coadd via interpolation, this should
        #   locate the PSF center at the proper pixel location in the final
        #   coadd
        dx = x - np.floor(x+0.5)
        dy = y - np.floor(y+0.5)

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

                # galsim can be very agressive
                self._galsim_psf_dim *= 5

            if isinstance(self._psf_model, galsim.InterpolatedImage):
                draw_method = 'no_pixel'
            else:
                draw_method = 'auto'

            im = self._psf_model.drawImage(
                nx=self._galsim_psf_dim,
                ny=self._galsim_psf_dim,
                wcs=wcs,
                offset=galsim.PositionD(x=dx, y=dy),
                method=draw_method,
            )
            psf_im = im.array.copy()

            if logging.DEBUG_PLOT >= logger.getEffectiveLevel():
                import matplotlib.pyplot as plt
                plt.figure()
                plt.title("SE PSF dx,dy = %s|%s" % (dx, dy))
                plt.imshow(psf_im)
                ax = plt.gca()
                ax.grid(False)
                plt.show()

        elif isinstance(self._psf_model, galsim.des.DES_PSFEx):
            # get the gs object
            psf = self._psf_model.getPSF(galsim.PositionD(
                x=x + self._wcs_position_offset,
                y=y + self._wcs_position_offset))

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
            # compute bounds in Piff wcs coords
            xmin, ymin = self._compute_psf_stamp_bounds(x, y, PIFF_STAMP_SIZE)
            xmin += self._wcs_position_offset
            ymin += self._wcs_position_offset
            bounds = galsim.BoundsI(
                xmin, xmin+PIFF_STAMP_SIZE-1,
                ymin, ymin+PIFF_STAMP_SIZE-1,
            )

            # draw into an image with these bounds
            # we leave the WCS unset so that piff
            # uses it's internal WCS for pixmappy
            # otherwise grab wcs
            if not isinstance(self._wcs, pixmappy.GalSimWCS):
                if self._wcs_type is not None:
                    assert self._wcs_type != "pixmappy", (
                        "WCS type cannot be pixmappy when using"
                        " wcs w/ Piff!"
                    )
                wcs = self.get_wcs_jacobian(x, y)
                image = galsim.ImageD(bounds, wcs=wcs)
            else:
                if self._wcs_type is not None:
                    assert self._wcs_type == "pixmappy", (
                        "WCS type must be pixmappy when using"
                        " no wcs w/ Piff!"
                    )
                image = galsim.ImageD(bounds)

            if psf_kwargs is not None:
                __kwargs = copy.deepcopy(self._psf_kwargs)
                __kwargs.update(psf_kwargs)
            else:
                __kwargs = self._psf_kwargs
            im = self._psf_model.draw(
                x=x + self._wcs_position_offset,
                y=y + self._wcs_position_offset,
                chipnum=self.source_info["ccdnum"],
                image=image,
                **__kwargs,
            )
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

    def compute_slice_bounds(self, ra, dec, box_size, frac_buffer):
        """Compute the patch bounds for a given ra,dec image, coadd box size,
        and frac buffer.

        Parameters
        ----------
        ra : float
            The right ascension of the sky position.
        dec : float
            The declination of the sky position.
        box_size : int
            The size of the square slice.
        frac_buffer : float
            The fractional amount by which to increase the coadd box size when computing
            the bounding box of the coadd grid in the SE image coords. This
            parameter can be used to account for position angle rotations by
            setting it up to sqrt(2) to account for full position angle rotations.
            In DES this number should be very close to 1.

        Returns
        -------
        patch_bounds : meds.bounds.Bounds
            The boundaries of the patch.
        """
        buff_box_size = int(frac_buffer * box_size)
        if buff_box_size % 2 != box_size % 2:
            buff_box_size += 1

        buff_box_cen = (buff_box_size - 1) / 2
        col, row = self.sky2image(ra, dec)
        se_start_row = int(np.floor(row - buff_box_cen + 0.5))
        se_start_col = int(np.floor(col - buff_box_cen + 0.5))
        patch_bnds = Bounds(
            rowmin=se_start_row,
            rowmax=se_start_row+buff_box_size-1,
            colmin=se_start_col,
            colmax=se_start_col+buff_box_size-1)

        return patch_bnds

    def set_psf(self, ra, dec, psf_kwargs=None):
        """Set the PSF of the slice using the input (ra, dec).

        Parameters
        ----------
        ra : float
            The right ascension of the sky position.
        dec : float
            The declination of the sky position.
        psf_kwargs : None or dict, optional
            If not None, the PSF will be drawn with these kwargs passed to the Piff
            call when using Piff.

        Returns
        -------
        flag : bool
            If True, the PSF model is flagged and this SE image at this point
            should not be used.
        """

        x, y = self.sky2image(ra, dec)

        if (
            self._mask_piff_failure_config is not None
            and isinstance(self._psf_model, piff.PSF)
        ):
            if psf_kwargs is not None:
                __kwargs = copy.deepcopy(self._psf_kwargs)
                __kwargs.update(psf_kwargs)
            else:
                __kwargs = self._psf_kwargs

            t0 = time.time()
            flag = _compute_bad_piff_model_delta_flag(
                    piff_path=self.source_info['piff_path'],
                    ccdnum=self.source_info['ccdnum'],
                    max_abs_t_diff=self._mask_piff_failure_config["max_abs_T_diff"],
                    grid_size=self._mask_piff_failure_config["grid_size"],
                    seed=self._mask_piff_failure_config["seed"],
                    piff_tuples=tuple((k, v) for k, v in __kwargs.items()),
                )
            if hasattr(_compute_bad_piff_model_delta_flag, "cache_info"):
                logger.debug(
                    'Piff bad mask cache info: %s',
                    _compute_bad_piff_model_delta_flag.cache_info()
                )
            logger.debug('Piff bad mask took %f seconds', time.time() - t0)
        else:
            flag = False

        psf = self.get_psf_image(x, y, psf_kwargs=psf_kwargs)
        xmin, ymin = self._compute_psf_stamp_bounds(x, y, psf.shape[0])

        self.psf = psf
        self.psf_x_start = xmin
        self.psf_y_start = ymin
        self.psf_box_size = psf.shape[0]

        return flag

    def set_interp_image_noise_pmask(self, *, interp_image, interp_noises, mask):
        """Set the inteprolated image, noise and processing mask.

        Parameters
        ----------
        interp_image : np.ndarray
            The interpolated image.
        interp_noises : list of np.ndarray
            The interpolated noise fields.
        mask : np.ndarray
            An array of ints w/ the same shape as the image indicating where
            interpolation was done.
        """
        msk = mask != 0

        self.interp_frac = np.zeros_like(self.image)
        self.interp_frac[msk] = 1.0

        self.orig_image = self.image
        self.interp_only_image = interp_image.copy()
        self.interp_only_image[~msk] = 0
        self.image = interp_image
        self.noises = interp_noises
        self.pmask = mask

    def map_image_by_nearest_pixel(
        self,
        *,
        image, x_start, y_start,
        wcs,
        wcs_position_offset,
        wcs_interp_delta
    ):
        """Map an image to the slice using the nearest pixel.

        NOTE: Typically, the input WCS object will be for a coadd.

        Parameters
        ----------
        image : np.ndarray
            The image to be mapped to the SE slice.
        x_start : int
            The zero-indexed, pixel-centered starting x/column of the input image.
        y_start : int
            The zero-indexed, pixel-centered starting y/row of the input image.
        wcs : `esutil.wcsutil.WCS` or `AffineWCS` object
            The WCS model to use for the input image. This object is
            assumed to take one-indexed, pixel-centered coordinates.
        wcs_position_offset : int
            The coordinate offset, if any, to get from the zero-indexed, pixel-
            centered coordinates used by this class to the coordinate convetion
            of the input `wcs` object.
        wcs_interp_delta : int
            The spacing in pixels used for interpolating the WCS transforms.
        """
        if not hasattr(self, 'box_size'):
            raise RuntimeError("You must call set_slice before mapping images!")

        mapped_image = np.zeros((self.box_size, self.box_size), dtype=image.dtype)

        t0 = time.time()
        _, wcs_interp = _get_wcs_inverse(
            wcs, wcs_position_offset,
            self,
            self._im_shape,
            wcs_interp_delta)
        if hasattr(_get_wcs_inverse, "cache_info"):
            logger.debug('wcs interp cache info: %s', _get_wcs_inverse.cache_info())
        logger.debug('wcs interp took %f seconds', time.time() - t0)

        y_self, x_self = np.mgrid[0:self.box_size, 0:self.box_size]
        x_self = x_self.ravel() + self.x_start
        y_self = y_self.ravel() + self.y_start
        x_img, y_img = wcs_interp(x_self, y_self)
        x_img -= x_start
        y_img -= y_start
        y_img = np.floor(y_img + 0.5).astype(np.int64)
        x_img = np.floor(x_img + 0.5).astype(np.int64)
        msk = (
            (x_img >= 0)
            & (x_img < image.shape[1])
            & (y_img >= 0)
            & (y_img < image.shape[0])
        )
        mapped_image[
            y_self[msk] - self.y_start,
            x_self[msk] - self.x_start
        ] = image[y_img[msk], x_img[msk]]
        return mapped_image

    def resample(
        self, *, wcs, wcs_position_offset, wcs_interp_shape,
        x_start, y_start, box_size,
        psf_x_start, psf_y_start, psf_box_size,
        se_wcs_interp_delta, coadd_wcs_interp_delta
    ):
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
        wcs : `esutil.wcsutil.WCS` or `AffineWCS` object
            The WCS model to use for the resampled pixel grid. This object is
            assumed to take one-indexed, pixel-centered coordinates.
        wcs_position_offset : int
            The coordinate offset, if any, to get from the zero-indexed, pixel-
            centered coordinates used by this class to the coordinate convetion
            of the input `wcs` object.
        wcs_interp_shape : tuple of ints
            The size of the box to be used to interpolate the wcs area
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
        se_wcs_interp_delta : int
            The spacing in pixels used to interpolate coadd pixel to SE pixel WCS
            function.
        coadd_wcs_interp_delta : int
            The spacing in pixels used for interpolating the coadd WCS pixel area.

        Returns
        -------
        resampled_data : dict
            A dictionary with the resampled data. It has keys

                'image' : the resampled image
                'bmask' : an approximate bmask using the nearest SE image
                    pixel
                'noises' : the resampled noise images
                'psf' : the resampled PSF image
                'pmask' : the resampled pmask
        """
        # error check
        if not hasattr(self, 'box_size'):
            raise RuntimeError("You must call set_slice before resmpling!")

        if not hasattr(self, 'psf_box_size'):
            raise RuntimeError("You must call set_psf before resmpling!")

        if not hasattr(self, 'pmask'):
            raise RuntimeError(
                "You must call set_interp_image_noise_pmask before resmpling!"
            )

        # 1. build the lookup table of SE position as a function of coadd
        # position and also table for area interpolation
        # we always go SE -> sky -> coadd because inverting the SE WCS will
        # be more expensive since they typically have distortion/tree ring
        # terms which require a root finder
        # we use a buffer to make sure edge pixels are ok
        t0 = time.time()
        wcs_interp, _ = _get_wcs_inverse(
            wcs, wcs_position_offset,
            self,
            self._im_shape,
            se_wcs_interp_delta)
        if hasattr(_get_wcs_inverse, "cache_info"):
            logger.debug('wcs interp cache info: %s', _get_wcs_inverse.cache_info())
        logger.debug('wcs interp took %f seconds', time.time() - t0)

        t0 = time.time()
        se_wcs_area_interp = _get_wcs_area_interp(
            self, self._im_shape, se_wcs_interp_delta
        )
        if hasattr(_get_wcs_area_interp, "cache_info"):
            logger.debug(
                'SE wcs area cache info: %s', _get_wcs_area_interp.cache_info()
            )
        logger.debug('SE wcs area interp took %f seconds', time.time() - t0)

        t0 = time.time()
        coadd_wcs_area_interp = _get_wcs_area_interp(
            wcs, wcs_interp_shape, coadd_wcs_interp_delta,
            position_offset=wcs_position_offset,
        )
        if hasattr(_get_wcs_area_interp, "cache_info"):
            logger.debug(
                'coadd wcs area cache info: %s', _get_wcs_area_interp.cache_info()
            )
        logger.debug('coadd wcs area interp took %f seconds', time.time() - t0)

        # 2. resample the PSF image
        resampled_data = {}
        resampled_data['psf'] = self.resample_psf(
            wcs=wcs,
            wcs_position_offset=wcs_position_offset,
            wcs_interp_shape=wcs_interp_shape,
            psf_x_start=psf_x_start,
            psf_y_start=psf_y_start,
            psf_box_size=psf_box_size,
            se_wcs_interp_delta=se_wcs_interp_delta,
            coadd_wcs_interp_delta=coadd_wcs_interp_delta,
        )

        # 3. using the lookup tables, we resample each image to the
        # coadd coordinates

        # compute the SE image positions using the lookup table
        y_rs, x_rs = np.mgrid[0:box_size, 0:box_size]
        y_rs = y_rs.ravel()
        x_rs = x_rs.ravel()
        # we need to add in the zero-indexed lower, left location of the
        # slice of the coadd image here since the WCS interp is in
        # absolute image pixel units
        x_rs_se, y_rs_se = wcs_interp(x_rs + x_start, y_rs + y_start)
        area_coadd = coadd_wcs_area_interp(x_rs + x_start, y_rs + y_start)

        # remove the offset to local coords for resampling
        x_rs_se -= self.x_start
        y_rs_se -= self.y_start

        y_self, x_self = np.mgrid[0:self.box_size, 0:self.box_size]
        x_self = x_self.ravel() + self.x_start
        y_self = y_self.ravel() + self.y_start
        area_se = se_wcs_area_interp(x_self, y_self)

        rim, edge = lanczos_resample(
            self.image / area_se.reshape(self.box_size, self.box_size),
            y_rs_se,
            x_rs_se,
        )
        rim *= area_coadd

        rns = []
        for _rn in self.noises:
            rn, _ = lanczos_resample(
                _rn / area_se.reshape(self.box_size, self.box_size),
                y_rs_se,
                x_rs_se,
            )
            rn *= area_coadd
            rns.append(rn)

        riom, rif, _ = lanczos_resample_two(
            self.interp_only_image / area_se.reshape(self.box_size, self.box_size),
            self.interp_frac / area_se.reshape(self.box_size, self.box_size),
            y_rs_se,
            x_rs_se,
        )
        riom *= area_coadd
        rif *= area_coadd

        if np.all(rim == 0):
            logger.warning("resampled SE image is all zero!")

        edge = edge.reshape(box_size, box_size)
        resampled_data.update({
            'image': rim.reshape(box_size, box_size),
            'noises': [rn.reshape(box_size, box_size) for rn in rns],
            'interp_only_image': riom.reshape(box_size, box_size),
            'interp_frac': rif.reshape(box_size, box_size),
            'edge': edge,
        })

        if logging.DEBUG_PLOT >= logger.getEffectiveLevel():
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("RESAMP SE image")
            plt.imshow(resampled_data["image"])
            ax = plt.gca()
            ax.grid(False)
            plt.show()

        # 4. do the nearest pixel for the bit mask
        y_rs_se = (y_rs_se + 0.5).astype(np.int64)
        x_rs_se = (x_rs_se + 0.5).astype(np.int64)
        msk = (
            (y_rs_se >= 0) & (y_rs_se < self.bmask.shape[0]) &
            (x_rs_se >= 0) & (x_rs_se < self.bmask.shape[1]))
        bmask = np.zeros((box_size, box_size), dtype=self.bmask.dtype)
        bmask[y_rs[msk], x_rs[msk]] = self.bmask[y_rs_se[msk], x_rs_se[msk]]
        bmask[y_rs[~msk], x_rs[~msk]] = BMASK_EDGE
        resampled_data['bmask'] = bmask

        # 5. do the nearest pixel for the pmask
        pmask = np.zeros((box_size, box_size), dtype=self.pmask.dtype)
        pmask[y_rs[msk], x_rs[msk]] = self.pmask[
            y_rs_se[msk], x_rs_se[msk]]
        pmask[y_rs[~msk], x_rs[~msk]] = BMASK_EDGE
        resampled_data['pmask'] = pmask
        resampled_data['pmask'][edge] |= BMASK_RESAMPLE_BOUNDS

        return resampled_data

    def resample_psf(
        self, *, wcs, wcs_position_offset, wcs_interp_shape,
        psf_x_start, psf_y_start, psf_box_size,
        se_wcs_interp_delta, coadd_wcs_interp_delta,
        dx=0, dy=0,
    ):
        """Resample a SEImageSlice PSF to a new WCS.

        The resampling is done as follows. For each pixel in the destination
        image, we compute its location in the source image. Then we use a
        Lanczos3 interpolant in the source grid to compute the value of the
        pixel in the destination image.

        NOTE: For destination grids that are far from the input grid, this
        function will not work.

        NOTE: Typically, the input WCS object will be for a coadd.

        NOTE: The parameters dx, dy are intended for use in correction for mean
        astrometric offsets due to color-dependent astrometry in coadd. The idea is
        that we have some default WCS color that is used everywhere when building
        coadds. We know this color is not correct for any given object which has some
        other color and thus a different astrometry and different PSF. We can correct
        for this by computing

            1) the PSF at objects color but with the WCS of the default color.
            2) the offset in SE pixels of a fixed location in ra,dec between the two
               colors

               dx, dy = (
                   WCS^(-1)(ra, dec, object color)
                   - WCS^(-1)(ra, dec, default color)
               )

        Then when we interpolate the SE PSF image to the coadd frame, we add -dx, -dy
        to the locations of the SE pixels computed with the default color. For a
        positive (dx, dy), this will shift the center of SE PSF by (dx, dy). Remember
        we shift the coordinates where we evaluate the image before coadding by
        -dx, -dy. The positive shift in the PSF center is the exact shift we
        (incorrectly) applied to the image when coadding originally (where we also
        evaluated the image at coordinates shifted by -dx, -dy). We now include it in
        the PSF center so that when we analyze the data with the PSF at the object's
        color, we get the right PSF size and extra braodening due to astrometric errors.

        Parameters
        ----------
        wcs : `esutil.wcsutil.WCS` or `AffineWCS` object
            The WCS model to use for the resampled pixel grid. This object is
            assumed to take one-indexed, pixel-centered coordinates.
        wcs_position_offset : int
            The coordinate offset, if any, to get from the zero-indexed, pixel-
            centered coordinates used by this class to the coordinate convetion
            of the input `wcs` object.
        wcs_interp_shape : tuple of ints
            The size of the box to be used to interpolate the wcs area
        psf_x_start : int
            The zero-indexed, pixel-centered starting x/column in the
            destination PSF grid.
        psf_y_start : int
            The zero-indexed, pixel-centered starting y/row in the
            destination PSF grid.
        psf_box_size : int
            The size of the square region to resample to in the destination
            PSF grid.
        se_wcs_interp_delta : int
            The spacing in pixels used to interpolate coadd pixel to SE pixel WCS
            function.
        coadd_wcs_interp_delta : int
            The spacing in pixels used for interpolating the coadd WCS pixel area.
        dx : float, optional
            An extra offset that is applied to the x SE image locations before
            resampling.
        dy : float, optional
            An extra offset that is applied to the y SE image locations before
            resampling.

        Returns
        -------
        psf : np.ndarray
            The resmapled PSF.
        """
        # error check
        if not hasattr(self, 'psf_box_size'):
            raise RuntimeError("You must call set_psf before resmpling!")

        # 1. build the lookup table of SE position as a function of coadd
        # position and also table for area interpolation
        # we always go SE -> sky -> coadd because inverting the SE WCS will
        # be more expensive since they typically have distortion/tree ring
        # terms which require a root finder
        # we use a buffer to make sure edge pixels are ok
        t0 = time.time()
        wcs_interp, _ = _get_wcs_inverse(
            wcs, wcs_position_offset,
            self,
            self._im_shape,
            se_wcs_interp_delta)
        if hasattr(_get_wcs_inverse, "cache_info"):
            logger.debug('wcs interp cache info: %s', _get_wcs_inverse.cache_info())
        logger.debug('wcs interp took %f seconds', time.time() - t0)

        t0 = time.time()
        se_wcs_area_interp = _get_wcs_area_interp(
            self, self._im_shape, se_wcs_interp_delta
        )
        if hasattr(_get_wcs_area_interp, "cache_info"):
            logger.debug(
                'SE wcs area cache info: %s', _get_wcs_area_interp.cache_info()
            )
        logger.debug('SE wcs area interp took %f seconds', time.time() - t0)

        t0 = time.time()
        coadd_wcs_area_interp = _get_wcs_area_interp(
            wcs, wcs_interp_shape, coadd_wcs_interp_delta,
            position_offset=wcs_position_offset,
        )
        if hasattr(_get_wcs_area_interp, "cache_info"):
            logger.debug(
                'coadd wcs area cache info: %s', _get_wcs_area_interp.cache_info()
            )
        logger.debug('coadd wcs area interp took %f seconds', time.time() - t0)

        # 2. do the PSF image
        y_rs, x_rs = np.mgrid[0:psf_box_size, 0:psf_box_size]
        y_rs = y_rs.ravel() + psf_y_start
        x_rs = x_rs.ravel() + psf_x_start
        x_rs_se, y_rs_se = wcs_interp(x_rs, y_rs)
        area_coadd = coadd_wcs_area_interp(x_rs, y_rs)

        y_self, x_self = np.mgrid[0:self.psf_box_size, 0:self.psf_box_size]
        x_self = x_self.ravel() + self.psf_x_start
        y_self = y_self.ravel() + self.psf_y_start
        area_se = se_wcs_area_interp(x_self, y_self)

        # remove the offset to local coords for resampling
        x_rs_se -= self.psf_x_start
        y_rs_se -= self.psf_y_start
        x_rs_se += dx
        y_rs_se += dy
        rs_psf, edge = lanczos_resample(
            self.psf / area_se.reshape(self.psf_box_size, self.psf_box_size),
            y_rs_se,
            x_rs_se,
        )
        rs_psf[edge] = 0
        rs_psf *= area_coadd
        resampled_psf = rs_psf.reshape(psf_box_size, psf_box_size)
        resampled_psf /= np.sum(resampled_psf)

        if logging.DEBUG_PLOT >= logger.getEffectiveLevel():
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("RESAMP PSF")
            plt.imshow(resampled_psf)
            ax = plt.gca()
            ax.grid(False)
            plt.show()

        return resampled_psf
