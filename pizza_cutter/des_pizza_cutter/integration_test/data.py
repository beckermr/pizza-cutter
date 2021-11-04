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


SIM_CONFIG = """\
fpack_pars:
  FZQVALUE: 4
  FZTILE: "(10240,1)"
  FZALGOR: "RICE_1"
  # preserve zeros, don't dither them
  FZQMETHD: "SUBTRACTIVE_DITHER_2"

coadd:
  # these are in pixels
  # the total "pizza slice" will be central_size + 2 * buffer_size
  central_size: 33  # size of the central region
  buffer_size: 8  # size of the buffer on each size

  psf_box_size: 51

  wcs_type: affine
  coadding_weight: 'noise'

single_epoch:
  # pixel spacing for building various WCS interpolants
  se_wcs_interp_delta: 8
  coadd_wcs_interp_delta: 8

  frac_buffer: 1
  psf_type: galsim
  psf_kwargs:
    a: null
  wcs_type: affine
  wcs_color: 0

  ignored_ccds:
   - 23

  reject_outliers: False
  symmetrize_masking: True
  copy_masked_edges: False
  max_masked_fraction: 0.1
  edge_buffer: 8

  mask_tape_bumps: False

  spline_interp_flags:
    - 2

  noise_interp_flags:
    - 4

  bad_image_flags:
    - 1
"""


SIM_CONFIG_ROTLIST = """\
fpack_pars:
  FZQVALUE: 4
  FZTILE: "(10240,1)"
  FZALGOR: "RICE_1"
  # preserve zeros, don't dither them
  FZQMETHD: "SUBTRACTIVE_DITHER_2"

coadd:
  # these are in pixels
  # the total "pizza slice" will be central_size + 2 * buffer_size
  central_size: 33  # size of the central region
  buffer_size: 8  # size of the buffer on each size

  psf_box_size: 51

  wcs_type: affine
  coadding_weight: 'noise'

single_epoch:
  # pixel spacing for building various WCS interpolants
  se_wcs_interp_delta: 8
  coadd_wcs_interp_delta: 8

  frac_buffer: 1
  psf_type: galsim
  psf_kwargs:
    a: null
  wcs_type: affine
  wcs_color: 0

  ignored_ccds:
   - 23

  reject_outliers: False
  symmetrize_masking: [90, 180, 270]
  copy_masked_edges: False
  max_masked_fraction: 0.2
  edge_buffer: 8

  mask_tape_bumps: False

  spline_interp_flags:
    - 2

  noise_interp_flags:
    - 4

  bad_image_flags:
    - 1
"""


SIM_CONFIG_GAIA = """\
fpack_pars:
  FZQVALUE: 4
  FZTILE: "(10240,1)"
  FZALGOR: "RICE_1"
  # preserve zeros, don't dither them
  FZQMETHD: "SUBTRACTIVE_DITHER_2"

coadd:
  # these are in pixels
  # the total "pizza slice" will be central_size + 2 * buffer_size
  central_size: 33  # size of the central region
  buffer_size: 8  # size of the buffer on each size

  psf_box_size: 51

  wcs_type: affine
  coadding_weight: 'noise'

single_epoch:
  # pixel spacing for building various WCS interpolants
  se_wcs_interp_delta: 8
  coadd_wcs_interp_delta: 8

  frac_buffer: 1
  psf_type: galsim
  psf_kwargs:
    a: null
  wcs_type: affine
  wcs_color: 0

  ignored_ccds:
   - 23

  reject_outliers: False
  symmetrize_masking: True
  copy_masked_edges: False
  max_masked_fraction: 0.1
  edge_buffer: 8

  mask_tape_bumps: False

  spline_interp_flags:
    - 2

  noise_interp_flags:
    - 4

  bad_image_flags:
    - 1

  gaia_star_masks:
    poly_coeffs: [1.4e-03, -1.6e-01,  3.5e+00]
    max_g_mag: 18.0
    symmetrize: False
"""


def write_sim(
    *, path, info, images, weights, bmasks, bkgs, gaia_stars=None,
):
    """Write an end-to-end sim to a path."""
    info_fname = expandpath(os.path.join(path, 'info.yaml'))
    makedir_fromfile(info_fname)

    tname = 'e2e-test'
    info['tilename'] = tname
    info['band'] = 'a'
    info['image_flags'] = 0
    info['magzp'] = MAGZP_REF
    info['scale'] = 1.0
    info['path'] = None
    info['filename'] = None

    info['image_path'] = "%s_a.fits.fz" % tname
    info['seg_path'] = "%s_a_segmap.fits" % tname
    info['bmask_path'] = "%s_a.fits.fz" % tname
    info['psf_path'] = "%s_a_psfcat.psf" % tname
    info['psfex_path'] = "%s_psfex.fits" % tname
    info['cat_path'] = "%s_a_cat.fits" % tname

    if gaia_stars is not None:
        info['gaia_stars_file'] = os.path.join(path, '%s_gaia-stars.fits' % tname)
        fitsio.write(info['gaia_stars_file'], gaia_stars, clobber=True)

    for ind, (image, weight, bmask, bkg) in enumerate(
            zip(images, weights, bmasks, bkgs)):

        info['src_info'][ind]['band'] = 'a'
        info['src_info'][ind]['expnum'] = 2*ind
        info['src_info'][ind]['ccdnum'] = 2*ind + 1
        info['src_info'][ind]['tilename'] = tname

        se_slug = "D%08d_%s_c%02d" % (2*ind, 'a', 2*ind+1)

        info['src_info'][ind]['piff_path'] = '%s_piff.fits' % se_slug
        info['src_info'][ind]['psfex_path'] = '%s_psfex.fits' % se_slug
        info['src_info'][ind]['psf_path'] = '%s_psf.fits' % se_slug
        info['src_info'][ind]['seg_path'] = '%s_seg.fits' % se_slug
        info['src_info'][ind]['head_path'] = "%s_%s_c%02d_scamp.ohead" % (
            tname, 'a', 2*ind+1,
        )
        info['src_info'][ind]['magzp'] = (
            MAGZP_REF - np.log10(info['src_info'][ind]['scale'])/0.4)

        fname = os.path.join(path, '%s_img%s.fits' % (se_slug, uuid.uuid4().hex))
        ext = '%s' % uuid.uuid4().hex[0:3]
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write(image, extname=ext)
        info['src_info'][ind]['image_path'] = fname
        info['src_info'][ind]['image_ext'] = ext

        info['src_info'][ind]['filename'] = os.path.basename(fname)
        info['src_info'][ind]['path'] = os.path.dirname(fname)

        fname = os.path.join(path, '%s_wgt%s.fits' % (se_slug, uuid.uuid4().hex))
        ext = '%s' % uuid.uuid4().hex[0:3]
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write(weight, extname=ext)
        info['src_info'][ind]['weight_path'] = fname
        info['src_info'][ind]['weight_ext'] = ext

        fname = os.path.join(path, '%s_bmask%s.fits' % (se_slug, uuid.uuid4().hex))
        ext = '%s' % uuid.uuid4().hex[0:3]
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write(bmask, extname=ext)
        info['src_info'][ind]['bmask_path'] = fname
        info['src_info'][ind]['bmask_ext'] = ext

        fname = os.path.join(path, '%s_bkg%s.fits' % (se_slug, uuid.uuid4().hex))
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
        image 11 - ignored CCD in config above

    This leaves 3 pristine images (images 12, 13, and 14) plus the ones
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
    wcs_var_fac = (1 + 2.0 * (rng.uniform(size=n_se_images) - 0.5) * 0.1)
    image_shape = (128, 256)
    buff = 8
    noises = 3.5e-4 + 1e-5 * np.arange(n_se_images)
    noise_fac = 1.0

    # randomly rotate the WCS
    thetas = (rng.uniform(size=n_se_images) - 0.5) * 2.0 * np.pi / 1e2
    dudxs = np.cos(thetas) * wcs_scale * wcs_var_fac
    dudys = -np.sin(thetas) * wcs_scale * wcs_var_fac
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
    gal = galsim.Gaussian(fwhm=2.5).shear(g1=-0.2, g2=0.5)

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
        ii['galsim_psf_config'] = {
            'type': 'Gaussian',
            'fwhm': psf_fwhm,
            'shear': {'type': 'G1G2', 'g1': 0.1, 'g2': -0.1}
        }
        ii['image_shape'] = list(image_shape)
        ii['image_flags'] = 0

        psf = galsim.Gaussian(fwhm=psf_fwhm).shear(g1=0.1, g2=-0.1)

        image = generate_input_se_image(
            gal=gal,
            psf=psf,
            image_shape=image_shape,
            wcs=AffineWCS(**ii['affine_wcs_config']),
            position_offset=position_offset,
            scale=scale)

        bkg = (np.random.normal(size=image.shape) * noise + 5 * noise) * noise_fac
        weight = np.zeros_like(image)
        weight[:, :] = 1.0 / noise / noise * scale * scale
        image += (rng.normal(size=image.shape) * noise / scale * noise_fac)
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
    x_loc = int(np.floor(x0s[3] + 0.5 - position_offsets[3]))
    y_loc = int(np.floor(y0s[3] + 0.5 - position_offsets[3]))
    bmasks[3][y_loc, x_loc] |= SIM_BMASK_BAD

    # image 4 is interpolated in a strip
    x_loc = int(np.floor(x0s[4] + 0.5 - position_offsets[4]))
    y_loc = int(np.floor(y0s[4] + 0.5 - position_offsets[4]))
    bmasks[4][y_loc:y_loc+2, :] |= SIM_BMASK_SPLINE_INTERP

    # image 5 is too masked
    bmasks[5][:, :] |= SIM_BMASK_SPLINE_INTERP

    # image 6 is too masked
    bmasks[6][:, :] |= SIM_BMASK_NOISE_INTERP

    # image 7 is too masked
    weights[7][:64, :] = 0

    # image 8 is noise interpolated
    x_loc = int(np.floor(x0s[8] + 0.5 - position_offsets[8]))
    y_loc = int(np.floor(y0s[8] + 0.5 - position_offsets[8]))
    bmasks[8][y_loc-10:y_loc-8, :] |= SIM_BMASK_NOISE_INTERP

    # image 10 is spline interpolated in the noise
    x_loc = int(np.floor(x0s[10] + 0.5 - position_offsets[10]))
    y_loc = int(np.floor(y0s[10] + 0.5 - position_offsets[10]))
    bmasks[10][y_loc-6:y_loc-4, :] |= SIM_BMASK_SPLINE_INTERP

    # set the coadd information too
    info['affine_wcs_config'] = {
        'dudx': wcs_scale,
        'dudy': 0,
        'dvdx': 0,
        'dvdy': wcs_scale,
        'x0': 24,
        'y0': 24}

    # the first number must match the "central_size" and the second must match
    # "buffer_size" in the config in in test_coadding_end2end.py those are
    # currently hard coded into the test, so you cannot change these

    info['image_shape'] = [33 + 2*8, 33 + 2*8]
    info['position_offset'] = 0

    # use image 3 to get reference wcs
    gaia_stars = generate_gaia_stars(
        rng=rng, image_shape=info['image_shape'], num=1,
        wcs=AffineWCS(**info['affine_wcs_config']),
    )
    return info, images, weights, bmasks, bkgs, gaia_stars


def generate_gaia_stars(rng, image_shape, num, wcs):
    """
    generate randomly placed gaia stars

    Parameters
    ----------
    rng: np.random.RandomState
        the rng for generating data
    image_shape: nrows, ncols
        Shape of image
    num: int
        Number of stars to generate
    wcs: galsim wcs
        Must have image2sky method
    """
    dt = [
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('phot_g_mean_mag', 'f4'),
        ('xgen', 'f8'),
        ('ygen', 'f8'),
    ]
    data = np.zeros(num, dtype=dt)

    # fairly small radii
    data['phot_g_mean_mag'] = 18

    rows = rng.uniform(low=10, high=image_shape[0]-10-1, size=num)
    cols = rng.uniform(low=10, high=image_shape[1]-10-1, size=num)
    data['xgen'] = cols
    data['ygen'] = rows

    data['ra'], data['dec'] = wcs.image2sky(cols, rows)
    return data


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
