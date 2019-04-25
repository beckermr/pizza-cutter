import time
import os
import copy
import logging
import shutil

logger = logging.getLogger(__name__)


def get_meds_base():
    """Get the base directory MEDS directory.

    Returns
    -------
    meds_dir : str
        The MEDS directory with any trailing slashes removed.
    """
    dr = copy.copy(os.environ['MEDS_DIR'])
    if dr[-1] == '/':
        dr = dir[0:-1]
    return dr


def get_meds_dir(medsconf, tilename):
    """Get the MEDS data directory for the input coadd tile.

    parameters
    ----------
    medsconf : str
        A name for the meds version or config.  e.g. 'y3a1-v02'
    tilename : str
        e.g. 'DES0417-5914'
    """
    bdir = get_meds_base()
    return os.path.join(bdir, medsconf, tilename)


def get_source_dir(medsconf, tilename, band):
    """Get the directory to hold input sources for MEDS files.

    Parameters
    ----------
    medsconf : str
        A name for the meds version or config.  e.g. 'y3a1-v02'
    tilename : str
        e.g. 'DES0417-5914'

    Returns
    -------
    dir : str
        The source directory.
    """
    dr = get_meds_dir(medsconf, tilename)
    return os.path.join(dr, 'sources-%s' % band)


def try_remove_timeout(fname, ntry=2, sleep_time=2):
    """Try and remove a filename with a timeout and retries.

    Parameters
    ----------
    fname : str
        The file to remove.
    ntry : int
        The number of retries.
    sleep_time : int
        The number of seconds to sleep between tries.
    """
    for i in range(ntry):
        try:
            os.remove(fname)
            break
        except Exception:
            if i == (ntry-1):
                raise
            else:
                print("could not remove '%s', trying again "
                      "in %f seconds" % (fname, sleep_time))
                time.sleep(sleep_time)


def expandpath(path):
    """Expand environment variables, user home directories (~), and convert
    to an absolute path.
    """
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    path = os.path.realpath(path)
    return path


def makedir_fromfile(fname):
    """Extract the directory and make it if it does not exist.
    """
    dname = os.path.dirname(fname)
    try_makedir(dname)


def try_makedir(dir):
    """Try to make the directory.
    """
    if not os.path.exists(dir):
        try:
            logger.debug("making directory: %s" % dir)
            os.makedirs(dir)
        except Exception:
            # probably a race condition
            pass


class StagedOutFile(object):
    """A context manager for staging files from temporary directories to
    a final destination.

    Parameters
    ----------
    fname : str
        Final destination path for file.
    tmpdir : str, optional
        If not sent, or `None`, the final path is used and no staging
        is performed.
    must_exist : bool, optional
        If `True`, the file to be staged must exist at the time of staging
        or an `IOError` is thrown. If `False`, this is silently ignored.
        Default `False`.

    Examples
    --------
    >>> fname = "/home/jill/output.dat"
    >>> tmpdir = "/tmp"
    >>> with StagedOutFile(fname, tmpdir=tmpdir) as sf:
    ...     with open(sf.path, 'w') as fobj:
    ...         fobj.write("some data")
    """
    def __init__(self, fname, tmpdir=None, must_exist=False):
        self.must_exist = must_exist
        self.was_staged_out = False
        self._set_paths(fname, tmpdir=tmpdir)

    def _set_paths(self, fname, tmpdir=None):
        fname = expandpath(fname)

        self.final_path = fname

        if tmpdir is not None:
            self.tmpdir = expandpath(tmpdir)
        else:
            self.tmpdir = tmpdir

        fdir = os.path.dirname(self.final_path)

        if self.tmpdir is None:
            self.is_temp = False
            self.path = self.final_path
        else:
            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)

            bname = os.path.basename(fname)
            self.path = os.path.join(self.tmpdir, bname)

            if self.tmpdir == fdir:
                # the user sent tmpdir as the final output dir, no
                # staging is performed
                self.is_temp = False
            else:
                self.is_temp = True

    def stage_out(self):
        """If a tempdir was used, move the file to its final destination.

        Note that you normally would not call this yourself, but rather use a
        context manager, in which case this method is called for you.
        """
        if self.is_temp and not self.was_staged_out:
            if not os.path.exists(self.path):
                if self.must_exist:
                    mess = "temporary file not found: %s" % self.path
                    raise IOError(mess)
                else:
                    return

            if os.path.exists(self.final_path):
                logger.debug("removing existing file: %s", self.final_path)
                os.remove(self.final_path)

            makedir_fromfile(self.final_path)

            logger.debug(
                "staging out '%s' -> '%s'", self.path, self.final_path)
            shutil.move(self.path, self.final_path)

        self.was_staged_out = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stage_out()

# import shutil
# import tarfile
# import yaml
# import tempfile
#
# try:
#     xrange
# except:
#     xrange = range
#
# def get_desdata():
#     """
#     get the environment variable DESDATA
#     """
#     dir=os.environ['DESDATA']
#     if dir[-1] == '/':
#         dir = dir[0:-1]
#     return dir
#
# def get_nwgint_config(campaign):
#     """
#     config used for making null weight images
#     """
#
#     clow = campaign.lower()
#     assert "y3a1" in clow or 'y3a2' in clow
#
#     dir=get_desdata()
#     path='OPS/config/multiepoch/Y3A1/v4/Y3A1_v4_coadd_nwgint.config'
#     path=os.path.join(dir, path)
#     return path
#
# def get_config_dir():
#     """
#     the config directory
#     """
#     if 'DESMEDS_CONFIG_DIR' not in os.environ:
#         raise RuntimeError("you need to define $DESMEDS_CONFIG_DIR")
#     return os.environ['DESMEDS_CONFIG_DIR']
#
# '''
# def get_list_dir():
#     """
#     directory holding lists and caches
#     """
#     desdata=get_desdata()
#     return os.path.join(desdata,'lists')
#
# def get_coadd_cache_file(campaign):
#     """
#     cache of coadd information for the given campaign
#
#     parameters
#     ----------
#     campaign: str
#         e.g. 'Y3A1_COADD'
#     """
#     dir=get_list_dir()
#     fname='%s-coadd-cache.fits' % campaign
#
#     fname = os.path.join(dir, fname)
#     return fname
#
# def get_coadd_src_cache_file(campaign):
#     """
#     cache of coadd source information for the given campaign
#
#     parameters
#     ----------
#     campaign: str
#         e.g. 'Y3A1_COADD'
#     """
#     dir=get_list_dir()
#     fname='%s-coadd-src-cache.fits' % campaign
#
#     fname = os.path.join(dir, fname)
#     return fname
#
# def get_zp_cache_file(campaign):
#     """
#     cache of coadd source information for the given campaign
#
#     parameters
#     ----------
#     campaign: str
#         e.g. 'Y3A1_COADD'
#     """
#     dir=get_list_dir()
#     fname='%s-zp-cache.fits' % campaign
#
#     fname = os.path.join(dir, fname)
#     return fname
# '''
#
# def get_meds_config_file(medsconf):
#     """
#     get the MEDS config file path
#
#     $DESMEDS_CONFIG_DIR must be defined
#
#     parameters
#     ----------
#     medsconf: str
#         Identifier for the meds config, e.g. "013"
#     """
#     dir=get_config_dir()
#     fname='meds-%s.yaml' % medsconf
#     return os.path.join(dir, fname)
#
# def get_tileset_file(tileset):
#     """
#     get the tileset yaml file
#
#     parameters
#     ----------
#     tileset: str
#         Identifier for the tileset, e.g. "y3-test01" or the
#         full path
#     """
#     dir=get_config_dir()
#     fname='tileset-%s.yaml' % tileset
#     return os.path.join(dir, fname)
#
#
# def read_meds_config(medsconf):
#     """
#     read the MEDS config file
#
#     $DESMEDS_CONFIG_DIR must be defined
#
#     parameters
#     ----------
#     medsconf: str
#         Identifier for the meds config, e.g. "013"
#     """
#
#     if '.yaml' in medsconf:
#         fname=medsconf
#         vers=os.path.basename(medsconf).replace('.yaml','').replace('meds-','')
#     else:
#         fname=get_meds_config_file(medsconf)
#         vers=medsconf
#
#     print("reading:",fname)
#     with open(fname) as fobj:
#         data=yaml.load(fobj)
#
#     if data['medsconf'] != vers:
#         raise ValueError("version mismatch: found '%s' rather "
#                          "than '%s'" % (data['medsconf'], vers))
#     return data
#
# def read_tileset(tileset):
#     """
#     read the tile set
#
#     parameters
#     ----------
#     tileset: str
#         Identifier for the tileset, e.g. "y3-test01" or the
#         full path
#     """
#
#     if '.yaml' in tileset:
#         fname=tileset
#     else:
#         fname=get_tileset_file(tileset)
#
#     print("reading:",fname)
#     with open(fname) as fobj:
#         data=yaml.load(fobj)
#
#     return data
#
#
# def get_testbed_config_file(testbed):
#     """
#     get the testbed config file path
#
#     $DESMEDS_CONFIG_DIR must be defined
#
#     parameters
#     ----------
#     testbed: str
#         Identifier for the testbed, e.g. "sva1-2"
#     """
#     dir=get_config_dir()
#     fname='testbed-%s.yaml' % testbed
#     return os.path.join(dir, fname)
#
# def read_testbed_config(testbed):
#     """
#     read the testbed configuration
#
#     $DESMEDS_CONFIG_DIR must be defined
#
#     parameters
#     ----------
#     testbed: str
#         Identifier for the testbed, e.g. "sva1-2"
#     """
#
#     fname=get_testbed_config_file(testbed)
#
#     print("reading:",fname)
#     with open(fname) as fobj:
#         data=yaml.load(fobj)
#
#     return data
#
# #
# # directories
# #
#
#
#
# def get_nullwt_dir(medsconf, tilename, band):
#     """
#     get the directory for the null weight image
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     """
#
#     dir=get_meds_dir(medsconf, tilename)
#     return os.path.join(dir, 'nullwt-%s' % band)
#
# def get_psf_dir(medsconf, tilename, band):
#     """
#     get the directory holding copies of the psf files
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'r'
#     """
#
#     dir=get_meds_dir(medsconf, tilename)
#     return os.path.join(dir, 'psfs-%s' % band)
#
# def get_lists_dir(medsconf, tilename, band):
#     """
#     get the directory holding the file lists and info
#     files
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'r'
#     """
#
#     dir=get_meds_dir(medsconf, tilename)
#     return os.path.join(dir, 'lists-%s' % band)
#
#
# def get_meds_script(medsconf, tilename, band):
#     """
#     get the meds script directory
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     ext='sh'
#     type='make-meds'
#     return get_meds_script_file_generic(medsconf, tilename, band, type, ext)
#
#
# def get_meds_script_dir(medsconf):
#     """
#     get the meds script directory
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     """
#
#     bdir = get_meds_base()
#     return os.path.join(bdir, medsconf, 'scripts')
#
#
# #
# # file paths
# #
#
# def get_meds_file(medsconf, tilename, band, ext='fits.fz'):
#     """
#     get the meds file for the input coadd run, band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='meds'
#     return get_meds_datafile_generic(medsconf,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
# def get_psfmap_file(medsconf, tilename, band):
#     """
#     get the meds psf map file
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='psfmap'
#     ext='dat'
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#     )
#
# def get_piff_map_file(medsconf, piff_run, tilename, band):
#     """
#     no longer used
#
#     psf map file for piff
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     dir=get_piff_map_dir(medsconf, piff_run, tilename, band)
#
#     fname = '%(tilename)s_%(band)s_psfmap-%(medsconf)s-%(piff_run)s.dat'
#     fname = fname % dict(
#         medsconf=medsconf,
#         piff_run=piff_run,
#         tilename=tilename,
#         band=band,
#     )
#     fname = os.path.join(dir, fname)
#     return fname
#
# def get_piff_map_dir(medsconf, piff_run, tilename, band):
#     """
#     psf map file for piff
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#     base_dir=os.environ['PIFF_MAP_DIR']
#     dir='%(base_dir)s/%(medsconf)s/%(piff_run)s/%(tilename)s'
#     dir = dir % dict(
#         base_dir=base_dir,
#         medsconf=medsconf,
#         piff_run=piff_run,
#         tilename=tilename,
#     )
#     return dir
#
#
# def get_piff_exp_summary_file(piff_run, expnum):
#     """
#     expnum not zero padded
#     """
#     base_dir=os.environ['PIFF_DATA_DIR']
#
#     fname = 'exp_psf_cat_%d.fits' % expnum
#
#     fname = os.path.join(
#         base_dir,
#         piff_run,
#         '%d' % expnum,
#         fname,
#     )
#     return fname
#
#
#
# def get_nullwt_file(medsconf, tilename, band, finalcut_file):
#     """
#     get the meds file for the input coadd run, band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     filalcut_file: str
#         name of original finalcut file
#     """
#
#     dir=get_nullwt_dir(medsconf, tilename, band)
#
#     # original will be something like
#     #   D00499389_r_c21_r2378p01_immasked.fits.fz
#     bname=os.path.basename(finalcut_file)
#     bname=bname.replace('.fits.fz','.fits')
#
#     fname=bname.replace('.fits','_nullwt.fits')
#
#     return os.path.join(dir, fname)
#
# def get_meds_stubby_file(medsconf, tilename, band):
#     """
#     get the stubby meds file, holding inputs for the MEDSMaker
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='meds-stubby'
#     ext='fits'
#     return get_meds_datafile_generic(medsconf,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
# def get_meds_stats_file(medsconf, tilename, band):
#     """
#     get the meds stats file for the input coadd run, band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='meds-stats'
#     ext='yaml'
#     return get_meds_datafile_generic(medsconf,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
# def get_meds_status_file(medsconf, tilename, band):
#     """
#     get the meds status file for the input
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='meds-status'
#     ext='yaml'
#     return get_meds_datafile_generic(medsconf,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
#
# def get_meds_srclist_file(medsconf, tilename, band):
#     """
#     get the meds source list file
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='meds-srclist'
#     ext='dat'
#     return get_meds_datafile_generic(medsconf,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
# def get_meds_input_file(medsconf, tilename, band):
#     """
#     get the meds input catalog file
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='meds-input'
#     ext='dat'
#     return get_meds_datafile_generic(medsconf,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
# def get_meds_coadd_objects_id_file(medsconf, coadd_run, band):
#     """
#     get the coadd objects id file for the input coadd run, band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     coadd_run: str
#         For SV and Y1, e.g. '20130828000021_DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     tilename=coadd_run_to_tilename(coadd_run)
#     type='meds-coadd-objects-id'
#     ext='dat'
#     return get_meds_datafile_generic(medsconf,
#                                      coadd_run,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
#
# def get_meds_datafile_generic(
#        medsconf, tilename, band, type, ext, subdir=None):
#     """
#     get the meds directory for the input tilename
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     type : str
#         e.g. 'meds' 'meds-stats' etc.
#     ext: str
#         extension, e.g. 'fits.fz' 'yaml' etc.
#     """
#
#     dir = get_meds_dir(medsconf, tilename)
#
#     if subdir is not None:
#         dir=os.path.join(dir, subdir)
#
#     # note bizarre pattern of dashes and underscores
#     # here we mimic DESDM
#
#     fname='%(tilename)s_%(band)s_%(type)s-%(medsconf)s.%(ext)s'
#     fname = fname % dict(tilename=tilename,
#                          band=band,
#                          type=type,
#                          medsconf=medsconf,
#                          ext=ext)
#     return os.path.join(dir, fname)
#
#
#
# def get_meds_lsf_file(medsconf, tilename, band, missing=False):
#     """
#     get the meds wq script file for the given tilename and band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config, e.g. 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     ext='lsf'
#     type='make-meds'
#     if missing:
#         type += '-missing'
#
#     return get_meds_script_file_generic(medsconf, tilename, band, type, ext)
#
# def get_meds_log_file(medsconf, tilename, band):
#     """
#     get the meds file for the input coadd run, band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     ext='log'
#     type='meds'
#     return get_meds_datafile_generic(medsconf,
#                                      tilename,
#                                      band,
#                                      type,
#                                      ext)
#
#
#
# def get_meds_wq_file(medsconf, tilename, band, missing=False):
#     """
#     get the meds wq script file for the given tilename and band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     ext='yaml'
#     type='make-meds'
#     if missing:
#         type += '-missing'
#
#     return get_meds_script_file_generic(medsconf, tilename, band, type, ext)
#
# def get_meds_stubby_wq_file(medsconf, tilename, band):
#     """
#     get the stubby meds wq script file for the given tilename and band
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y1a1-v01'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     ext='yaml'
#     type='make-stubby'
#     return get_meds_script_file_generic(medsconf, tilename, band, type, ext)
#
#
#
# def get_meds_script_file_generic(medsconf, tilename, band, type, ext):
#     """
#     get the meds script maker file for the given tilename and band
#
#     parameters
#     ----------
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     type: str
#         any string
#     ext: str
#         extension, e.g. 'sh' 'yaml'
#     """
#     dir=get_meds_script_dir(medsconf)
#
#     fname = '%(tilename)s-%(band)s-%(type)s.%(ext)s'
#     fname = fname % dict(tilename=tilename,
#                          band=band,
#                          type=type,
#                          ext=ext)
#
#     return os.path.join(dir, fname)
#
#
# class StagedInFile(object):
#     """
#     A class to represent a staged file
#     If tmpdir=None no staging is performed and the original file path is used
#
#     parameters
#     ----------
#     fname: str
#         original file location
#     tmpdir: str, optional
#         If not sent, no staging is done.
#
#     examples
#     --------
#     # using a context for the staged file
#     fname="/home/jill/output.dat"
#     tmpdir="/tmp"
#     with StagedInFile(fname,tmpdir=tmpdir) as sf:
#         with open(sf.path) as fobj:
#             # read some data
#
#     """
#     def __init__(self, fname, tmpdir=None):
#
#         self._set_paths(fname, tmpdir=tmpdir)
#         self.stage_in()
#
#     def _set_paths(self, fname, tmpdir=None):
#         fname=expandpath(fname)
#
#         self.original_path = fname
#
#         if tmpdir is not None:
#             self.tmpdir = expandpath(tmpdir)
#         else:
#             self.tmpdir = tmpdir
#
#         self.was_staged_in = False
#         self._stage_in = False
#
#         if self.tmpdir is not None:
#             bdir,bname = os.path.split(self.original_path)
#             self.path = os.path.join(self.tmpdir, bname)
#
#             if self.tmpdir == bdir:
#                 # the user sent tmpdir as the source dir, no
#                 # staging is performed
#                 self._stage_in = False
#             else:
#                 self._stage_in = True
#
#     def stage_in(self):
#         """
#         make a local copy of the file
#         """
#         import shutil
#
#         if self._stage_in:
#             if not os.path.exists(self.original_path):
#                 raise IOError("file not found:",self.original_path)
#
#             if os.path.exists(self.path):
#                 print("removing existing file:",self.path)
#                 os.remove(self.path)
#             else:
#                 makedir_fromfile(self.path)
#
#             print("staging in",self.original_path,"->",self.path)
#             shutil.copy(self.original_path,self.path)
#
#             self.was_staged_in = True
#
#     def cleanup(self):
#         if os.path.exists(self.path) and self.was_staged_in:
#             print("removing temporary file:",self.path)
#             os.remove(self.path)
#             self.was_staged_in = False
#
#     def __del__(self):
#         self.cleanup()
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exception_type, exception_value, traceback):
#         self.cleanup()
#
#
# class StagedOutFile(object):
#     """
#     A class to represent a staged file
#     If tmpdir=None no staging is performed and the original file
#     path is used
#     parameters
#     ----------
#     fname: str
#         Final destination path for file
#     tmpdir: str, optional
#         If not sent, or None, the final path is used and no staging
#         is performed
#     must_exist: bool, optional
#         If True, the file to be staged must exist at the time of staging
#         or an IOError is thrown. If False, this is silently ignored.
#         Default False.
#     examples
#     --------
#
#     fname="/home/jill/output.dat"
#     tmpdir="/tmp"
#     with StagedOutFile(fname,tmpdir=tmpdir) as sf:
#         with open(sf.path,'w') as fobj:
#             fobj.write("some data")
#
#     """
#     def __init__(self, fname, tmpdir=None, must_exist=False):
#
#         self.must_exist = must_exist
#         self.was_staged_out = False
#
#         self._set_paths(fname, tmpdir=tmpdir)
#
#
#     def _set_paths(self, fname, tmpdir=None):
#         fname=expandpath(fname)
#
#         self.final_path = fname
#
#         if tmpdir is not None:
#             self.tmpdir = expandpath(tmpdir)
#         else:
#             self.tmpdir = tmpdir
#
#         fdir = os.path.dirname(self.final_path)
#
#         if self.tmpdir is None:
#             self.is_temp = False
#             self.path = self.final_path
#         else:
#             if not os.path.exists(self.tmpdir):
#                 os.makedirs(self.tmpdir)
#
#             bname = os.path.basename(fname)
#             self.path = os.path.join(self.tmpdir, bname)
#
#             if self.tmpdir==fdir:
#                 # the user sent tmpdir as the final output dir, no
#                 # staging is performed
#                 self.is_temp = False
#             else:
#                 self.is_temp = True
#
#     def stage_out(self):
#         """
#         if a tempdir was used, move the file to its final destination
#         note you normally would not call this yourself, but rather use a
#         context, in which case this method is called for you
#         with StagedOutFile(fname,tmpdir=tmpdir) as sf:
#             #do something
#         """
#         import shutil
#
#         if self.is_temp and not self.was_staged_out:
#             if not os.path.exists(self.path):
#                 if self.must_exist:
#                     mess = "temporary file not found: %s" % self.path
#                     raise IOError(mess)
#                 else:
#                     return
#
#             if os.path.exists(self.final_path):
#                 print("removing existing file:",self.final_path)
#                 os.remove(self.final_path)
#
#             makedir_fromfile(self.final_path)
#
#             print("staging out '%s' -> '%s'" % (self.path,self.final_path))
#             shutil.move(self.path,self.final_path)
#
#         self.was_staged_out=True
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exception_type, exception_value, traceback):
#         self.stage_out()
#
# class TempFile(object):
#     """
#     A class to represent a temporary file
#
#     parameters
#     ----------
#     fname: str
#         The full path for file
#
#     examples
#     --------
#
#     # using a context for the staged file
#     fname="/home/jill/output.dat"
#     with TempFile(fname) as sf:
#         with open(sf.path,'w') as fobj:
#             fobj.write("some data")
#
#             # do something with the file
#     """
#     def __init__(self, fname):
#         self.path = fname
#
#         self.was_cleaned_up = False
#
#     def cleanup(self):
#         """
#         remove the file if it exists, if not already cleaned up
#         """
#         import shutil
#
#         if not self.was_cleaned_up:
#             if os.path.exists(self.path):
#                 print("removing:",self.path)
#                 os.remove(self.path)
#
#             self.was_cleaned_up=True
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exception_type, exception_value, traceback):
#         self.cleanup()
#
#
# def expandpath(path):
#     """
#     expand environment variables, user home directories (~), and convert
#     to an absolute path
#     """
#     path=os.path.expandvars(path)
#     path=os.path.expanduser(path)
#     path=os.path.realpath(path)
#     return path
#
#
# def makedir_fromfile(fname):
#     """
#     extract the directory and make it if it does not exist
#     """
#     dname=os.path.dirname(fname)
#     try_makedir(dname)
#
# def try_makedir(dir):
#     """
#     try to make the directory
#     """
#     if not os.path.exists(dir):
#         try:
#             print("making directory:",dir)
#             os.makedirs(dir)
#         except:
#             # probably a race condition
#             pass
#
# def get_temp_dir():
#     """
#     get a temporary directory.  Check for batch system specific
#     directories in environment variables, falling back to TMPDIR
#     """
#     tmpdir=os.environ.get('_CONDOR_SCRATCH_DIR',None)
#     if tmpdir is None:
#         tmpdir=os.environ.get('TMPDIR',None)
#         if tmpdir is None:
#             tmpdir = tempfile.mkdtemp()
#     return tmpdir
#
#
# def read_yaml(fname):
#     with open(fname) as fobj:
#         data=yaml.load(fobj)
#
#     return data
#
#
# #
# # specific for the desdm version
# #
#
# def get_desdm_file_config(medsconf, tilename, band):
#     """
#     the desdm version needs a file config
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='fileconf'
#     ext='yaml'
#     subdir='lists-%s' % band
#
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#         subdir=subdir,
#     )
#
# def get_desdm_finalcut_flist(medsconf, tilename, band):
#     """
#     the desdm version needs a list
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='finalcut-flist'
#     ext='dat'
#     subdir='lists-%s' % band
#
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#         subdir=subdir,
#     )
#
#
# def get_desdm_nullwt_flist(medsconf, tilename, band):
#     """
#     the desdm version needs a list
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='nullwt-flist'
#     ext='dat'
#     subdir='lists-%s' % band
#
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#         subdir=subdir,
#     )
#
# def get_coaddinfo_file(medsconf, tilename, band):
#     """
#     the desdm version needs a list
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='coaddinfo'
#     ext='yaml'
#     subdir='lists-%s' % band
#
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#         subdir=subdir,
#     )
#
#
# def get_desdm_seg_flist(medsconf, tilename, band):
#     """
#     the desdm version needs a list
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='seg-flist'
#     ext='dat'
#     subdir='lists-%s' % band
#
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#         subdir=subdir,
#     )
#
#
# def get_desdm_bkg_flist(medsconf, tilename, band):
#     """
#     the desdm version needs a list
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='bkg-flist'
#     ext='dat'
#     subdir='lists-%s' % band
#
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#         subdir=subdir,
#     )
#
#
# def get_desdm_objmap(medsconf, tilename, band):
#     """
#     the desdm version needs a map
#
#     parameters
#     ----------
#     medsconf: str
#         A name for the meds version or config.  e.g. '013'
#         or 'y3a1-v02'
#     tilename: str
#         e.g. 'DES0417-5914'
#     band: str
#         e.g. 'i'
#     """
#
#     type='objmap'
#     ext='fits'
#     subdir='lists-%s' % band
#
#     return get_meds_datafile_generic(
#         medsconf,
#         tilename,
#         band,
#         type,
#         ext,
#         subdir=subdir,
#     )
#
# def try_remove_timeout(fname, ntry=2, sleep_time=2):
#     import time
#
#     for i in xrange(ntry):
#         try:
#             os.remove(fname)
#             break
#         except:
#             if i==(ntry-1):
#                 raise
#             else:
#                 print("could not remove '%s', trying again "
#                       "in %f seconds" % (fname,sleep_time))
#                 time.sleep(sleep_time)
#
# def try_remove(f):
#     try:
#         os.remove(f)
#         print("removed file:",f)
#     except:
#         print("could not remove file:",f)
#
#
# def try_remove_dir(d):
#     try:
#         shutil.rmtree(d)
#         print("removed dir:",d)
#     except:
#         print("could not remove dir:",d)
#
#
# def tar_directory(source_dir):
#     """
#     tar a directory to a tar file called directory.tar.gz
#     """
#     outfile=source_dir+'.tar.gz'
#     print("tarring directory %s -> %s" % (source_dir, outfile))
#     with tarfile.open(outfile, "w:gz") as tar:
#         tar.add(source_dir, arcname=os.path.basename(source_dir))
