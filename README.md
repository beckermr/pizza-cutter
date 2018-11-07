# pizza-cutter

yummy survey slices!

## Running `metadetect` on a "pizza slice" MEDS File

The module `pizza_cutter.metadetect` has code to run `metadetect`
on "pizza slice" MEDS files. You can use the command line tool 
`run-metadetect-on-slices` like this

```bash
 run-metadetect-on-slices \
     --config=<metadetect config> \
     --seed=<seed> \
     MEDS_FILE_1 MEDS_FILE_2
```

## Processing the DES Deblending Sims from Niall

The module `pizza_cutter.coadd_sim_slicer` has code to convert Niall's
deblending sims to "pizza slice" MEDS files. Follow the following steps.

0. Install this package.
1. Make the sims (or possibly download them from NERSC) and put them in a
  directory, say `sims`.
2. Get the appropriate PSFEX `.psf` file from DESDM for the coadd tile and band.
  Put this file in `sims` directory.
3. Write a config file. See the [example](config_files/cmc-griz-005-flat.yaml).
4. Invoke the command line tool to make the "pizza slice" MEDS file:

```bash
coadd-sim-pizza-cutter \
    --config=config.yaml \
    --sim-path=sims \
    --tilename=<tilename> \
    --band=<band> \
    --tag=<tag, optional>
```
