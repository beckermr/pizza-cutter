import os
import numpy as np
import pytest

import galsim
import piff
import pixmappy

from .._se_image import SEImageSlice


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'SEImageSlice can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize('wcs_pos_offset', [0, 1])
def test_se_image_wcs_color_pixmappy(se_image_data, wcs_pos_offset):
    psf = piff.PSF.read(se_image_data["piff_data"]["g"])
    if isinstance(psf.wcs[0], pixmappy.GalSimWCS):
        wcs = psf.wcs[0]

        # HACK at the internals to code around a bug!
        if isinstance(
            wcs.origin,
            galsim._galsim.PositionD
        ):
            wcs._origin = galsim.PositionD(
                wcs._origin.x,
                wcs._origin.y,
            )
    else:
        raise RuntimeError("No pixmappy WCS available to test with!")

    se_im = SEImageSlice(
        source_info=se_image_data['source_info'],
        psf_model=None,
        wcs=wcs,
        wcs_position_offset=wcs_pos_offset,
        wcs_color=0.7,
        psf_kwargs=None,
        noise_seeds=[10],
        mask_tape_bumps=False,
    )

    x = np.arange(2)
    y = np.arange(2) + 100

    ra, dec = se_im.image2sky(x, y)
    x_out, y_out = se_im.sky2image(ra, dec)

    for i in range(2):
        pos = wcs.toWorld(
            galsim.PositionD(x=x[i]+wcs_pos_offset, y=y[i]+wcs_pos_offset),
            color=0.7,
        )
        assert np.allclose(ra[i], pos.ra / galsim.degrees, atol=1e-16, rtol=0)
        assert np.allclose(dec[i], pos.dec / galsim.degrees, atol=1e-16, rtol=0)

        pos = wcs.toWorld(
            galsim.PositionD(x=x[i]+wcs_pos_offset, y=y[i]+wcs_pos_offset),
            color=0.0,
        )
        assert not np.allclose(ra[i], pos.ra / galsim.degrees, atol=1e-16, rtol=0)
        assert not np.allclose(dec[i], pos.dec / galsim.degrees, atol=1e-16, rtol=0)

        wpos = galsim.CelestialCoord(
            ra=ra[i] * galsim.degrees,
            dec=dec[i] * galsim.degrees,
        )
        pos = wcs.toImage(wpos, color=0.7)
        assert np.allclose(x[i], pos.x - wcs_pos_offset, atol=1e-8, rtol=0)
        assert np.allclose(y[i], pos.y - wcs_pos_offset, atol=1e-8, rtol=0)

        pos = wcs.toImage(wpos, color=0.0)
        assert not np.allclose(x[i], pos.x - wcs_pos_offset, atol=1e-8, rtol=0)
        assert not np.allclose(y[i], pos.y - wcs_pos_offset, atol=1e-8, rtol=0)

        dx = x[i] - (pos.x - wcs_pos_offset)
        dy = y[i] - (pos.y - wcs_pos_offset)
        dr = np.sqrt(dx**2 + dy**2) * 0.263  # approx pixel scale for DES in arcsec
        assert dr > 0.010
        assert dr < 0.100
        print("\ndcr shift in arcsec:", dr, flush=True)

        jac = se_im.get_wcs_jacobian(x[i], y[i])
        jac_col = wcs.local(
            image_pos=galsim.PositionD(x=x[i]+wcs_pos_offset, y=y[i]+wcs_pos_offset),
            color=0.7,
        )
        jac_nocol = wcs.local(
            image_pos=galsim.PositionD(x=x[i]+wcs_pos_offset, y=y[i]+wcs_pos_offset),
            color=0,
        )
        assert jac == jac_col
        assert jac != jac_nocol
        print("\njacobians:\n", jac, "\n", jac_col, "\n", jac_nocol, flush=True)

        area = se_im.get_wcs_pixel_area(x[i], y[i])
        assert np.allclose(area, jac_col.pixelArea())
        assert not np.allclose(area, jac_nocol.pixelArea())
        print(
            "\nareas:\n", area,
            "\n", jac_col.pixelArea(),
            "\n", jac_nocol.pixelArea(),
            flush=True,
        )
