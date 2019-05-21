"""Code to run integration tests for the pizza cutter.

We are going to coadd a Gaussian blob with various WCS models and
masking effects.

Things the code does:

    1. We will define a few noise bits to check various code features

        SIM_BMASK_BAD            2**0  # images that have a bad pixel
        SIM_BMASK_SPLINE_INTERP  2**1  # pixels get spline interpolated
        SIM_BMASK_NOISE_INTERP   2**2  # pixels get noise interpolated

    2. We add images with large masked fractions.

    3. We add images with image_flags set in the input.

    4. We make sure some images do not fully overlap the coadd region.

    5. We add images that would overlap except for the edge buffer.

Note that this test is not meant to cover every possible combination of paths
in the code, but to verify it works once.
"""

import os
import uuid
import yaml
import numpy as np

import galsim
import fitsio

from .._affine_wcs import AffineWCS
from ...files import expandpath, makedir_fromfile
from .._constants import MAGZP_REF

SIM_BMASK_BAD = 2**0  # images that have a bad pixel
SIM_BMASK_SPLINE_INTERP = 2**1  # pixels get spline interpolated
SIM_BMASK_NOISE_INTERP = 2**2  # pixels get noise interpolated


def write_sim(*, path, info, images, weights, bmasks, bkgs):
    """Write an end-to-end sim to a path."""
    info_fname = expandpath(os.path.join(path, 'info.yaml'))
    makedir_fromfile(info_fname)

    info['tilename'] = 'e2e_test'
    info['band'] = 'p'
    info['image_flags'] = 0
    info['magzp'] = MAGZP_REF
    info['scale'] = 1.0

    for ind, (image, weight, bmask, bkg) in enumerate(
            zip(images, weights, bmasks, bkgs)):

        info['src_info'][ind]['magzp'] = (
            MAGZP_REF - np.log10(info['src_info'][ind]['scale'])/0.4)

        fname = os.path.join(path, 'img%s.fits' % uuid.uuid4().hex)
        ext = '%s' % uuid.uuid4().hex[0:3]
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write(image, extname=ext)
        info['src_info'][ind]['image_path'] = fname
        info['src_info'][ind]['image_ext'] = ext

        info['src_info'][ind]['filename'] = os.path.basename(fname)

        fname = os.path.join(path, 'wgt%s.fits' % uuid.uuid4().hex)
        ext = '%s' % uuid.uuid4().hex[0:3]
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write(weight, extname=ext)
        info['src_info'][ind]['weight_path'] = fname
        info['src_info'][ind]['weight_ext'] = ext

        fname = os.path.join(path, 'bmask%s.fits' % uuid.uuid4().hex)
        ext = '%s' % uuid.uuid4().hex[0:3]
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write(bmask, extname=ext)
        info['src_info'][ind]['bmask_path'] = fname
        info['src_info'][ind]['bmask_ext'] = ext

        fname = os.path.join(path, 'bkg%s.fits' % uuid.uuid4().hex)
        ext = '%s' % uuid.uuid4().hex[0:3]
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write(bkg, extname=ext)
        info['src_info'][ind]['bkg_path'] = fname
        info['src_info'][ind]['bkg_ext'] = ext

    with open(info_fname, 'w') as fp:
        yaml.dump(info, fp)


def generate_sim():
    """Generate a set of SE images and their metadata for coadding.

    The returned images have various defects.

        image 0 - doesn't overlap the coadding region
        image 1 - has 'image_flags' set to 10
        image 2 - coadding region intersects the edge of the image
        image 3 - a pixel with SIM_BMASK_BAD set
        image 4 - interpolated in a strip through the object
        image 5 - too high mask fraction (SIM_BMASK_SPLINE_INTERP)
        image 6 - too high mask fraction (SIM_BMASK_NOISE_INTERP)
        image 7 - too high mask fraction (weight = 0)
        image 8 - noise interpolated on the edge
        image 10 - spline interpolated on the edge

    This leaves 4 pristine images (images 11, 12, 13, and 14) plus the ones
    that can be used from the list above (images 4, 8, 9, and 10).

    Returns
    -------
    info : dict
        A dictionary with the information about the images in the
        correct format for coadding.
    images : list of np.ndarray
        The images to coadd.
    weights : list of np.ndarray
        The weight maps of the imags to coadd.
    bmasks : list of np.ndarray
        The bit masks of the images to coadd.
    bkgs : list of np.ndarray
        The background images.
    """
    seed = 11
    n_se_images = 15
    rng = np.random.RandomState(seed=seed)

    wcs_scale = 0.25
    image_shape = (128, 256)
    buff = 8
    noises = 3.5e-4 + 1e-5 * np.arange(n_se_images)

    # randomly rotate the WCS
    thetas = (rng.uniform(size=n_se_images) - 0.5) * 2.0 * np.pi / 1e2
    dudxs = np.cos(thetas) * wcs_scale
    dudys = -np.sin(thetas) * wcs_scale
    dvdxs = -dudys.copy()
    dvdys = dudxs.copy()

    # randomly flip the axes
    sign_xs = rng.choice([-1, 1], size=n_se_images, replace=True)
    sign_ys = rng.choice([-1, 1], size=n_se_images, replace=True)
    dudxs *= sign_xs
    dvdxs *= sign_xs
    dudys *= sign_ys
    dvdys *= sign_ys

    # and displace the center
    ebuff = 8 + 16 + buff
    x0s = rng.uniform(
        size=n_se_images,
        low=ebuff,
        high=image_shape[1]-ebuff)
    y0s = rng.uniform(
        size=n_se_images,
        low=ebuff,
        high=image_shape[0]-ebuff)

    # image zero doesn't intersect
    x0s[0] = image_shape[1] * 10
    y0s[0] = -image_shape[0] * 10

    # image 2 is on the edge
    x0s[2] = 10
    y0s[2] = image_shape[0] - 10

    # use weird coordinate conventions
    position_offsets = rng.choice(
        [-2, -1.5, -0.5, 0, 0.5, 1.5, 2], size=n_se_images, replace=True)
    x0s += position_offsets
    y0s += position_offsets

    # and random image scales
    scales = rng.uniform(size=n_se_images) * 0.6 + 0.4

    info = {}
    src_info = []
    images = []
    weights = []
    bmasks = []
    bkgs = []

    psf_fwhms = 0.7 + 0.1*rng.uniform(size=n_se_images)
    gal = galsim.Gaussian(fwhm=0.5).shear(g1=0, g2=0.5)

    for (dudx, dudy, dvdx, dvdy, x0, y0, position_offset,
         scale, noise, psf_fwhm) in zip(
            dudxs, dudys, dvdxs, dvdys, x0s, y0s, position_offsets,
            scales, noises, psf_fwhms):
        ii = {}
        ii['affine_wcs_config'] = {
            'dudx': float(dudx),
            'dudy': float(dudy),
            'dvdx': float(dvdx),
            'dvdy': float(dvdy),
            'x0': float(x0),
            'y0': float(y0)}
        ii['scale'] = float(scale)
        ii['position_offset'] = float(position_offset)
        ii['galsim_psf_config'] = {'type': 'Gaussian', 'fwhm': psf_fwhm}
        ii['image_shape'] = list(image_shape)
        ii['image_flags'] = 0

        psf = galsim.Gaussian(fwhm=psf_fwhm)

        image = generate_input_se_image(
            gal=gal,
            psf=psf,
            image_shape=image_shape,
            wcs=AffineWCS(**ii['affine_wcs_config']),
            position_offset=position_offset,
            scale=scale)

        bkg = np.random.normal(size=image.shape) * noise + 5 * noise
        weight = np.zeros_like(image)
        weight[:, :] = 1.0 / noise / noise * scale * scale
        image += (rng.normal(size=image.shape) * noise / scale)

        image += bkg

        images.append(image)
        weights.append(weight)
        bmasks.append(np.zeros(image.shape, dtype=np.int32))
        bkgs.append(bkg)
        src_info.append(ii)

    info['src_info'] = src_info

    # image 1 has flags
    info['src_info'][1]['image_flags'] = 10
    images[1][:, :] = np.nan

    # image 3 has a bad pixel that totally excludes it
    x_loc = int(x0s[3] + 0.5 - position_offsets[3])
    y_loc = int(y0s[3] + 0.5 - position_offsets[3])
    bmasks[3][y_loc, x_loc] |= SIM_BMASK_BAD

    # image 4 is interpolated in a strip
    x_loc = int(x0s[4] + 0.5 - position_offsets[4])
    y_loc = int(y0s[4] + 0.5 - position_offsets[4])
    bmasks[4][y_loc:y_loc+2, :] |= SIM_BMASK_SPLINE_INTERP

    # image 5 is too masked
    bmasks[5][:, :] |= SIM_BMASK_SPLINE_INTERP

    # image 6 is too masked
    bmasks[6][:, :] |= SIM_BMASK_NOISE_INTERP

    # image 7 is too masked
    weights[7][:64, :] = 0

    # image 8 is noise interpolated
    x_loc = int(x0s[8] + 0.5 - position_offsets[8])
    y_loc = int(y0s[8] + 0.5 - position_offsets[8])
    bmasks[8][y_loc-10:y_loc-8, :] |= SIM_BMASK_NOISE_INTERP

    # image 10 is spline interpolated in the noise
    x_loc = int(x0s[10] + 0.5 - position_offsets[10])
    y_loc = int(y0s[10] + 0.5 - position_offsets[10])
    bmasks[10][y_loc-6:y_loc-4, :] |= SIM_BMASK_SPLINE_INTERP

    # set the coadd information too
    info['affine_wcs_config'] = {
        'dudx': wcs_scale,
        'dudy': 0,
        'dvdx': 0,
        'dvdy': wcs_scale,
        'x0': 24,
        'y0': 24}
    info['image_shape'] = [33 + 2*8, 33 + 2*8]
    info['position_offset'] = 0

    return info, images, weights, bmasks, bkgs


def generate_input_se_image(
        *, gal, psf, image_shape, wcs, position_offset, scale):
    """Generate input SE image data.

    NOTE: This function only works for Affine or flat WCS solutions with a
    constant pixel scale and shape.

    Parameters
    ----------
    gal : galsim.GSObject
        the galaxy as a galsim object.
    psf : galsim.GSObject
        The PSF as a galsim object.
    image_shape : tuple
        The output shape of the image.
    wcs : AffineWCS
        The image WCS as an AffineWCS object.
    position_offset : float
        The offset to convert from zero-indexed, pixel-centered coordinates
        to the convention used by the WCS.
    scale : float
        The desired factor by which to rescale the SE image to the common
        output scale. The returned image is properly normalized via
        multiplying by the input scale.

    Returns
    -------
    image : np.nadarray
        The output image.
    """
    psf_plus_gal = galsim.Convolve(gal, psf)
    gs_wcs = galsim.AffineTransform(
        dudx=wcs.dudx,
        dudy=wcs.dudy,
        dvdx=wcs.dvdx,
        dvdy=wcs.dvdy,
        origin=galsim.PositionD(x=wcs.x0, y=wcs.y0))

    # we have to offset the object in the image to correspond to the correct
    # position in WCS (u,v) coordinates (which is always (0, 0) by convention
    # for this test).
    image_pos = gs_wcs.toImage(galsim.PositionD(x=0, y=0))

    # canonical image center w/ position offset
    image_cen = galsim.PositionD(
        x=(image_shape[1] - 1)/2 + position_offset,
        y=(image_shape[0] - 1)/2 + position_offset)

    # the offset is the difference between these two
    # note that galsim will interpret this as the offset in a one-indexed
    # coordinate system, but that is OK. It is the difference that
    # matters here.
    offset = image_pos - image_cen

    image = psf_plus_gal.drawImage(
        nx=image_shape[1],
        ny=image_shape[0],
        wcs=gs_wcs.local(image_pos=image_pos),
        offset=offset)

    return image.array / scale
