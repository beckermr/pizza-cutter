import numpy as np
import galsim

from .._affine_wcs import AffineWCS


def generate_sim(*, seed, n_se_images=5):
    """Generate a set of SE images to coadd.


    """
    rng = np.random.RandomState(seed=seed)

    scale = 0.25
    image_shape = (128, 64)
    buff = 8

    # randomly rotate the WCS
    thetas = (rng.uniform(size=n_se_images) - 0.5) * 2.0 * np.pi / 1e2
    dudxs = np.cos(thetas) * scale
    dudys = -np.sin(thetas) * scale
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
    x0s = rng.uniform(
        size=n_se_images,
        low=buff,
        high=image_shape[1]-buff)
    y0s = rng.uniform(
        size=n_se_images,
        low=buff,
        high=image_shape[0]-buff)

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

    psf_fwhm = 0.7
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    gal = galsim.Gaussian(fwhm=0.5).shear(g1=0, g2=0.5)

    for dudx, dudy, dvdx, dvdy, x0, y0, position_offset, scale in zip(
            dudxs, dudys, dvdxs, dvdys, x0s, y0s, position_offsets, scales):
        ii = {}
        ii['affine_wcs_config'] = {
            'dudx': dudx,
            'dudy': dudy,
            'dvdx': dvdx,
            'dvdy': dvdy,
            'x0': x0,
            'y0': y0}
        ii['scale'] = scale
        ii['position_offset'] = position_offset
        ii['galsim_psf_config'] = {'type': 'Gaussian', 'fwhm': psf_fwhm}
        ii['image_shape'] = list(image_shape)
        ii['image_flags'] = 0

        images.append(generate_input_se_image(
            gal=gal,
            psf=psf,
            image_shape=image_shape,
            wcs=AffineWCS(**ii['affine_wcs_config']),
            position_offset=position_offset,
            scale=scale))
        src_info.append(ii)

    info['src_info'] = src_info

    # image 1 has flags
    info['src_info'][1]['image_flags'] = 10
    images[1][:, :] = np.nan

    return info, images


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
