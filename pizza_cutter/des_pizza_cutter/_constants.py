# flags
BMASK_EDGE = 2**29
BMASK_NOISE_INTERP = 2**28
BMASK_SPLINE_INTERP = 2**27

# kernel extends outside bounds in resampling
BMASK_RESAMPLE_BOUNDS = 2**26

# bit associated with gaia star mask
BMASK_GAIA_STAR = 2**25

# these are constants that are etched in stone for MEDS files
MAGZP_REF = 30.0
OBJECT_DATA_EXTNAME = 'object_data'
IMAGE_INFO_EXTNAME = 'image_info'
METADATA_EXTNAME = 'metadata'
EPOCHS_INFO_EXTNAME = 'epochs_info'
GAIA_STARS_EXTNAME = 'gaia_stars'

IMAGE_CUTOUT_EXTNAME = 'image_cutouts'
WEIGHT_CUTOUT_EXTNAME = 'weight_cutouts'
SEG_CUTOUT_EXTNAME = 'seg_cutouts'
BMASK_CUTOUT_EXTNAME = 'bmask_cutouts'
ORMASK_CUTOUT_EXTNAME = 'ormask_cutouts'
NOISE_CUTOUT_EXTNAME = 'noise_cutouts'
MFRAC_CUTOUT_EXTNAME = 'mfrac_cutouts'
PSF_CUTOUT_EXTNAME = 'psf'
CUTOUT_DTYPES = {
    'image_cutouts': 'f4',
    'weight_cutouts': 'f4',
    'seg_cutouts': 'i4',
    'bmask_cutouts': 'i4',
    'ormask_cutouts': 'i4',
    'noise_cutouts': 'f4',
    'psf': 'f4',
    'mfrac_cutouts': 'f4',
}
CUTOUT_DEFAULT_VALUES = {
    'image_cutouts': 0.0,
    'weight_cutouts': 0.0,
    'seg_cutouts': 0,
    'bmask_cutouts': BMASK_EDGE,
    'ormask_cutouts': BMASK_EDGE,
    'noise_cutouts': 0.0,
    'psf': 0.0,
    'mfrac_cutouts': 0.0,
}

# this is always true for the DES
POSITION_OFFSET = 1
