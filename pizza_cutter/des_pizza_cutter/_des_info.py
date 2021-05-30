import logging
import os
import subprocess

from ._constants import MAGZP_REF, POSITION_OFFSET
from ._piff_tools import load_piff_path_from_image_path

logger = logging.getLogger(__name__)


def add_extra_des_coadd_tile_info(*, info, piff_run):
    """Read the coadd tile info, load WCS info, and load PSF info for
    the DES Y3+ DESDM layout.

    Parameters
    ----------
    info: dict
        Info dict for a coadd tile
    piff_run : str
        The PIFF PSF run to use.

    Returns
    -------
    info : dict
        A dictionary with at least the following keys:

            'position_offset' : the offset to add to zero-indexed image
                coordinates to get transform them to the convention assumed
                by the WCS.
            'src_info' : list of dicts for the SE sources
            'image_path' : the path to the FITS file with the coadd image
            'image_ext' : the name of the FITS extension with the coadd image
            'weight_path' : the path to the FITS file with the coadd weight map
            'weight_ext' : the name of the FITS extension with the coadd weight
                map
            'bmask_path' : the path to the FITS file with the coadd bit mask
            'bmask_ext' : the name of the FITS extension with the coadd bit
                mask
            'seg_path' : the path to the FITS file with the coadd seg map
            'seg_ext' : the name of the FITS extension with the coadd seg map
            'image_flags' : any flags for the coadd image
            'scale' : a multiplicative factor to apply to the image
                (`*= scale`) and weight map (`/= scale**2`) for magnitude
                zero-point calibration.
            'magzp' : the magnitude zero point for the image

        The dictionaries in the 'src_info' list have at least the
        following keys:

            'image_path' : the path to the FITS file with the SE image
            'image_ext' : the name of the FITS extension with the SE image
            'bkg_path' : the path to the FITS file with the SE background image
            'bkg_ext' : the name of the FITS extension with the SE background
                image
            'weight_path' : the path to the FITS file with the SE weight map
            'weight_ext' : the name of the FITS extension with the SE weight
                map
            'bmask_path' : the path to the FITS file with the SE bit mask
            'bmask_ext' : the name of the FITS extension with the SE bit mask
            'psfex_path' : the path to the PSFEx PSF model
            'piff_path' : the path to the Piff PSF model
            'scale' : a multiplicative factor to apply to the image
                (`*= scale`) and weight map (`/= scale**2`) for magnitude
                zero-point calibration
            'magzp' : the magnitude zero point for the image
            'image_flags' : any flags for the SE image
    """

    info['position_offset'] = POSITION_OFFSET

    info['image_ext'] = 'sci'

    info['weight_path'] = info['image_path']
    info['weight_ext'] = 'wgt'

    info['bmask_path'] = info['image_path']
    info['bmask_ext'] = 'msk'

    info['seg_ext'] = 'sci'

    # always true for the coadd
    info['magzp'] = MAGZP_REF
    info['scale'] = 1.0
    info['image_shape'] = [10000, 10000]

    info['image_flags'] = 0  # TODO set this properly for the coadd?

    for index, ii in enumerate(info['src_info']):
        ii['image_shape'] = [4096, 2048]
        ii['image_flags'] = 0

        ii['image_ext'] = 'sci'

        ii['weight_path'] = ii['image_path']
        ii['weight_ext'] = 'wgt'

        ii['bmask_path'] = ii['image_path']
        ii['bmask_ext'] = 'msk'

        ii['bkg_ext'] = 'sci'

        # wcs info
        ii['position_offset'] = POSITION_OFFSET

        # psfex psf
        ii['psfex_path'] = ii['psf_path']

        # piff
        if piff_run is not None:
            piff_data = load_piff_path_from_image_path(
                image_path=ii['image_path'],
                piff_run=piff_run,
            )

            if piff_data['psf_path'] is not None:
                ii['piff_path'] = piff_data['psf_path']
            ii['image_flags'] |= piff_data['flags']

        # image scale
        ii['scale'] = 10.0**(0.4*(MAGZP_REF - ii['magzp']))


def check_info(*, info):
    """Do some sanity checks on the info data. Will raise if something
    fails.

    Parameters
    ----------
    info : dict
        The diction built by the `des-pizza-cutter-prep-tile` command.
    """
    errors = []

    # 1: check that coadd paths appear to be for the same coadd tile
    tilenames = set()
    coadd_keys = [
        "image_path", "seg_path", "bmask_path",
        "gaia_stars_file", "psf_path",
    ]
    for key in coadd_keys:
        fname = os.path.basename(info[key]).split("_")[0]
        tilenames |= set([fname])
    if len(tilenames) > 1:
        errors.append(
            "The coadd files are not for the same tile! %s" % (
                [info[k] for k in coadd_keys]
            )
        )
    tilename = list(tilenames)[0]

    # 2: coadd always has scale 1
    if info["scale"] != 1.0:
        errors.append("The coadd image scale (%s) is not 1.0!" % info["scale"])

    # 3: bands should match
    bands = set()
    bands |= set([info["band"]])
    for ii in info["src_info"]:
        bands |= set([ii["band"]])
    if len(bands) > 1:
        errors.append(
            "The band entries do not all match! %s" % bands
        )
    band = list(bands)[0]

    # 4: bands in coadd file names should match band above
    ends = dict(
        bmask_path=f"_{band}.fits.fz",
        cat_path=f"_{band}_cat.fits",
        image_path=f"_{band}.fits.fz",
        psf_path=f"_{band}_psfcat.psf",
        seg_path=f"_{band}_segmap.fits",
    )
    for key, end in ends.items():
        if not info[key].endswith(end):
            errors.append(
                "File path %s doesn't end with %s and this looks wrong!" % (
                    info[key], end,
                )
            )

    # 5: make sure filenames in info exts match band, ccdnum and expnum
    se_keys = [
        "bkg_path",
        "bmask_path",
        "image_path",
        "piff_path",
        "psfex_path",
        "psf_path",
        "seg_path",
        "weight_path",
    ]
    for ii in info["src_info"]:
        ccd_slug = "D%08d_%s_c%02d_" % (ii["expnum"], band, ii["ccdnum"])
        for key in se_keys:
            if not os.path.basename(ii[key]).startswith(ccd_slug):
                errors.append(
                    "File path %s doesn't start with %s and this looks wrong!" % (
                        ii[key], ccd_slug,
                    )
                )

    # 6: make sure tilename is consistent
    for ii in info["src_info"]:
        if ii["tilename"] != tilename:
            errors.append("SE source %s has the wrong tilename! Should be %s!" % (
                ii, tilename,
            ))

    # 7: make sure scamp head starts with tilename and ends with right slug
    for ii in info["src_info"]:
        if not os.path.basename(ii["head_path"]).startswith(tilename):
            errors.append("File path %s doesn't start with %s and this looks wrong!" % (
                ii["head_path"], tilename,
            ))

        scamp_slug = "_%s_c%02d_scamp.ohead" % (band, ii["ccdnum"])
        if not os.path.basename(ii["head_path"]).endswith(scamp_slug):
            errors.append("File path %s doesn't end with %s and this looks wrong!" % (
                ii["head_path"], scamp_slug,
            ))

    # report and raise any errors
    if len(errors) > 0:
        print(
            "found errors:\n\n===\n\n%s\n\n===\n\n"
            % "\n\n===\n\n".join(errors),
            flush=True,
        )
        raise RuntimeError(
            "Found problems with info file for tile!\n\n===\n\n%s\n\n===\n\n"
            % "\n\n===\n\n".join(errors)
        )


def get_gaia_path(tilename, version="v1"):
    """Get the path in the filearchive for the GAIA catalog for this tile.

    Parameters
    ----------
    tilename : str
        The name of the coadd tile (e.g., "DES0146-3623").
    version : str, optional
        The GAIA catalog version. Defaults to "v1"

    Returns
    -------
    archive_path : str
        The path in the DESDM file archive (e.g.,
        "OPS/cal/cat_tile_gaia/v1/DES0146-3623_GAIA_DR2_v1.fits").
    """
    import easyaccess as ea
    conn = ea.connect(section='desoper')
    curs = conn.cursor()
    curs.execute("""
select
    fai.path, fai.filename
from
    desfile d,
    file_archive_info fai
where
    d.filetype = 'cat_tile_gaia'
    and d.filename like '%s%%%s.fits'
    and d.filename = fai.filename
""" % (tilename, version))
    c = curs.fetchall()
    if len(c) == 0:
        raise RuntimeError("No GAIA file can be found for tile '%s'!" % tilename)
    pth, fname = c[0]
    return os.path.join(pth, fname)


def download_archive_file(archive_path, source_dir):
    """Given a path in the DESDM file archive and the destination directory,
    download the file via rsync.

    Parameters
    ----------
    archive_path : str
        The file to download from the DESDM file archive (e.g.,
        "OPS/cal/cat_tile_gaia/v1/DES0146-3623_GAIA_DR2_v1.fits").
    source_dir : str
        The location to download the file to. The file will be at
        `source_dir`/`archive_path`.
    """

    if 'DESREMOTE_RSYNC_USER' in os.environ:
        user = os.environ['DESREMOTE_RSYNC_USER'] + '@'
    else:
        user = ""

    final_dir = os.path.dirname(os.path.join(source_dir, archive_path))
    os.makedirs(final_dir, exist_ok=True)

    rsync_cmd = """\
rsync \
    -av \
    --password-file ${DES_RSYNC_PASSFILE} \
    %(user)s${DESREMOTE_RSYNC}/%(fname)s \
    %(source_dir)s/%(fname)s
""" % dict(
        user=user,
        fname=archive_path,
        source_dir=source_dir,
    )

    subprocess.run(
        rsync_cmd,
        shell=True,
        check=True,
    )
