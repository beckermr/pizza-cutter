#!/usr/bin/env python
import os
import sys
import tempfile
import logging

from pizza_cutter.files import StagedOutFile
from pizza_cutter.simulation.des_masks import gen_masks_from_des_y3_images
from pizza_cutter.des_pizza_cutter._des_coadd_data import (
    DESCoadd, DESCoaddSources)
from pizza_cutter.files import expandpath

MEDSCONF = 'y3a1-v02'
CAMPAIGN = 'Y3A1_COADD'
TILENAMES = ['DES0417-5914']

logging.basicConfig(stream=sys.stdout)
logging.getLogger('pizza_cutter').setLevel(logging.INFO)


def _gen_for_band(band, seed):

    bmask_paths = []
    for tilename in TILENAMES:
        se_srcs = DESCoaddSources(
            medsconf=MEDSCONF,
            campaign=CAMPAIGN,
            tilename=tilename,
            band=band)
        coadd = DESCoadd(
            medsconf=MEDSCONF,
            campaign=CAMPAIGN,
            tilename=tilename,
            band=band,
            sources=se_srcs)

        info = coadd.get_info()
        for src in info['src_info']:
            pth = expandpath(os.path.join(
                os.environ['MEDS_DIR'],
                MEDSCONF,
                tilename,
                'sources-%s' % band,
                src['path'],
                src['filename']))
            if 'compression' in src and len(src['compression']) > 0:
                pth += src['compression']
            bmask_paths.append(pth)

        coadd.download()

    output_file = 'des_y3_bit_mask_cat_%s.fits' % band
    with tempfile.TemporaryDirectory() as tmpdir:
        with StagedOutFile(output_file + '.fz', tmpdir) as sf:
            gen_masks_from_des_y3_images(
                bmask_paths=bmask_paths,
                bmask_ext='msk',
                output_file=sf.path.replace('.fz', ''),
                n_per=16,
                seed=seed,
                nrows=500,
                ncols=500)


bands = ['g', ]  # 'r', 'i', 'z']
seeds = [1, ]  # 45, 56, 78]

for band, seed in zip(bands, seeds):
    _gen_for_band(band, seed)
