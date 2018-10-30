import os
import shutil
import copy
import tempfile
import subprocess

from . import files

_DOWNLOAD_CMD = r"""
    rsync \
        -av \
        --password-file $DES_RSYNC_PASSFILE \
        --files-from=%(flist_file)s \
        %(userstring)s${DESREMOTE_RSYNC}/ \
        %(source_dir)s/
"""

_QUERY_COADD_TEMPLATE_BYTILE = """\
select
    m.tilename as tilename,
    fai.path as path,
    fai.filename as filename,
    fai.compression as compression,
    m.band as band,
    m.pfw_attempt_id as pfw_attempt_id

from
    prod.proctag t,
    prod.coadd m,
    prod.file_archive_info fai
where
    t.tag='%(campaign)s'
    and t.pfw_attempt_id=m.pfw_attempt_id
    and m.tilename='%(tilename)s'
    and m.band='%(band)s'
    and m.filetype='coadd'
    and fai.filename=m.filename
    and fai.archive_name='desar2home'\n"""

_QUERY_COADD_SRC_BYTILE = """\
select
    i.tilename,
    fai.path,
    j.filename as filename,
    fai.compression,
    j.band as band,
    i.pfw_attempt_id,
    z.mag_zero as magzp
from
    image i,
    image j,
    proctag tme,
    proctag tse,
    file_archive_info fai,
    zeropoint z
where
    tme.tag='%(campaign)s'
    and tme.pfw_attempt_id=i.pfw_attempt_id
    and i.filetype='coadd_nwgint'
    and i.tilename='%(tilename)s'
    and i.expnum=j.expnum
    and i.ccdnum=j.ccdnum
    and j.filetype='red_immask'
    and j.pfw_attempt_id=tse.pfw_attempt_id
    and j.band='%(band)s'
    and tse.tag='%(finalcut_campaign)s'
    and fai.filename=j.filename
    and z.imagename = j.filename
    and z.source='FGCM'
    and z.version='v2.0'
order by
    filename
"""


class DESCoadd(object):
    """A tool for working with DES coadds and their sources.

    Parameters
    ----------
    medsconf : str
        A name for the meds version or config.  e.g. 'y3a1-v02'
    tilename: str
        A tile name, e.g. 'DES0417-5914'.
    band : str
        A character gicing the band, e.g., 'r'.
    campaign : str, optional
        The coadd campaign, e.g., 'Y3A1_COADD'.
    sources : `DESCoaddSources` instance, optional
        An instantiated `DESCoaddSources` object.

    Attributes
    ----------
    source_dir : str
        The directory that coadds and possibly their sources will
        be downloaded to.

    Methods
    -------
    get_info : Returns information about this coadd and possibly its sources.
    download : Dowloads the coadd and possibly its sources to `source_dir`.
    clean : Removes any downloaded files in `source_dir`.
    """
    def __init__(self, medsconf, tilename, band,
                 campaign='Y3A1_COADD', sources=None):

        self.medsconf = medsconf
        self.tilename = tilename
        self.band = band

        self.source_dir = files.get_source_dir(
            self.medsconf,
            self.tilename,
            self.band)

        self.campaign = campaign.upper()
        self.sources = sources

    def get_info(self):
        """Get the coadd info for the tile and band.

        If sources were set, add source info as well.

        Returns
        -------
        info : dict
            A dictionary with coadd info and possibly source info.
        """

        if hasattr(self, '_info'):
            info = self._info
        else:
            info = self._do_query()

            # add full path info
            self._add_full_paths(info)

            if self.sources is not None:
                self._add_src_info(info)

            self._info = info

        return info

    def download(self):
        """Download the sources for a tile and band.
        """

        if not os.path.exists(self.source_dir):
            print("making source dir:", self.source_dir)
            os.makedirs(self.source_dir)

        info = self.get_info()

        self.flist_file = self._write_download_flist(info)

        if 'DESREMOTE_RSYNC_USER' in os.environ:
            userstring = os.environ['DESREMOTE_RSYNC_USER'] + '@'
        else:
            userstring = ''

        cmd = _DOWNLOAD_CMD % {
            'flist_file': self.flist_file,
            'userstring': userstring,
            'source_dir': self.source_dir}

        try:
            subprocess.check_call(cmd, shell=True)
        finally:
            files.try_remove_timeout(self.flist_file)

        return info

    def clean(self):
        """Remove downloaded files for the specified tile and band.
        """
        print("removing sources:", self.source_dir)
        shutil.rmtree(self.source_dir)

    def _do_query(self):
        """Get info for the specified tilename and band
        """

        query = _QUERY_COADD_TEMPLATE_BYTILE % {
            'campaign': self.campaign,
            'tilename': self.tilename,
            'band': self.band
        }

        print('executing query:\n', query)
        conn = self.get_conn()
        curs = conn.cursor()
        curs.execute(query)

        c = curs.fetchall()

        tile, path, fname, comp, band, pai = c[0]

        entry = {
            'tilename': tile,
            'filename': fname,
            'compression': comp,
            'path': path,
            'band': band,
            'pfw_attempt_id': pai,

            # need to add this to the cache?  should always
            # be the same...
            'magzp': 30.0,
        }

        return entry

    def _add_full_paths(self, info):
        """Adjust the image paths to match the full path.

        Note: Seg maps don't have .fz extension for coadd.
        """
        dirdict = self._get_all_dirs(info)

        info['image_path'] = os.path.join(
            dirdict['image']['local_dir'],
            info['filename']+info['compression'],
        )
        info['cat_path'] = os.path.join(
            dirdict['cat']['local_dir'],
            info['filename'].replace('.fits', '_cat.fits'),
        )
        info['seg_path'] = os.path.join(
            dirdict['seg']['local_dir'],
            info['filename'].replace('.fits', '_segmap.fits'),
        )
        info['psf_path'] = os.path.join(
            dirdict['psf']['local_dir'],
            info['filename'].replace('.fits', '_psfcat.psf'),
        )

    def _get_download_flist(self, info, no_prefix=False):
        """Get list of files for this tile.

        Parameters
        ----------
        info: dict
            The info dict for this tile/band, possibly including
            the src_info

        no_prefix: bool
            If True, the {source_dir} is removed from the front
        """
        source_dir = copy.copy(self.source_dir)

        if source_dir[-1] != '/':
            source_dir += '/'

        types = self._get_download_types()
        stypes = self._get_source_download_types()

        flist = []
        for tpe in types:
            tname = '%s_path' % tpe

            fname = info[tname]

            if no_prefix:
                fname = fname.replace(source_dir, '')

            flist.append(fname)

        if 'src_info' in info:
            for sinfo in info['src_info']:
                for tpe in stypes:
                    tname = '%s_path' % tpe

                    fname = sinfo[tname]

                    if no_prefix:
                        fname = fname.replace(source_dir, '')

                    flist.append(fname)

        return flist

    def _write_download_flist(self, info):

        flist_file = self._get_tempfile()
        flist = self._get_download_flist(info, no_prefix=True)

        print("writing file list to:", flist_file)
        with open(flist_file, 'w') as fobj:
            for fname in flist:
                fobj.write(fname)
                fobj.write('\n')

        return flist_file

    def _get_tempfile(self):
        return tempfile.mktemp(
            prefix='coadd-flist-',
            suffix='.dat',
        )

    def _get_download_types(self):
        return ['image', 'cat', 'seg', 'psf']

    def _get_source_download_types(self):
        return ['image', 'bkg', 'seg', 'psf', 'head']

    def _add_src_info(self, info):
        """Get path info for the input single-epoch sources.
        """

        src_info = self.sources.get_info()
        self._add_head_full_paths(info, src_info)
        info['src_info'] = src_info

    def _add_head_full_paths(self, info, src_info):
        dirdict = self._get_all_dirs(info)

        # this is a full path
        auxdir = dirdict['aux']['local_dir']
        head_front = info['filename'][0:-7]

        for src in src_info:
            fname = src['filename']
            fid = fname[0:15]
            head_fname = '%s_%s_scamp.ohead' % (head_front, fid)
            src['head_path'] = os.path.join(
                auxdir,
                head_fname,
            )

    def get_conn(self):
        if not hasattr(self, '_conn'):
            self._make_conn()

        return self._conn

    def _make_conn(self):
        import easyaccess as ea
        conn = ea.connect(section='desoper')
        self._conn = conn

    def _get_all_dirs(self, info):
        dirs = {}

        path = info['path']
        dirs['image'] = self._get_dirs(path)
        dirs['cat'] = self._get_dirs(path, type='cat')
        dirs['aux'] = self._get_dirs(path, type='aux')
        dirs['seg'] = self._get_dirs(path, type='seg')
        dirs['psf'] = self._get_dirs(path, type='psf')

        return dirs

    def _get_dirs(self, path, type=None):
        """Get directories for downloads given an input path.

        Parameters
        ----------
        path : str
            The input path, e.g.
            "OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/coadd/".
        type : str, optional
            The type of path. For example, if the input path is

                OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/coadd/

            then setting `type` would yield

                OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/{type}/

        Returns
        -------
        paths : dict
            Dictionary with keys
                'local_dir': the local directory
                'remote_dir': the remote directory
        """
        local_dir = '%s/%s' % (self.source_dir, path)
        remote_dir = '$DESREMOTE_RSYNC/%s' % path

        local_dir = os.path.expandvars(local_dir)
        remote_dir = os.path.expandvars(remote_dir)

        if type is not None:
            local_dir = self._extract_alt_dir(local_dir, type)
            remote_dir = self._extract_alt_dir(remote_dir, type)

        return {
            'local_dir': local_dir,
            'remote_dir': remote_dir}

    def _extract_alt_dir(self, path, type):
        """Extract the catalog path from an image path.

        For example, setting `path` to

            OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/coadd/

        would yield

            OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/{type}/

        for whatever string is in type.
        """
        ps = path.split('/')
        assert ps[-1] == 'coadd'

        ps[-1] = type
        return '/'.join(ps)


class DESCoaddSources(object):
    """A tool to work with DES coadd sources (SE images, etc.).

    Parameters
    ----------
    medsconf : str
        A name for the meds version or config.  e.g. 'y3a1-v02'
    tilename: str
        A tile name, e.g. 'DES0417-5914'.
    band : str
        A character gicing the band, e.g., 'r'.
    campaign : str, optional
        The coadd campaign, e.g., 'Y3A1_COADD'.

    Attributes
    ----------
    source_dir : str
        The directory that coadds and possibly their sources will
        be downloaded to.
    finalcut_campaign : str
        The finalcut campaign corresponding to the input coadd campaign.

    Methods
    -------
    get_info : Returns information about this coadd and possibly its sources.
    """
    def __init__(self, medsconf, tilename, band,
                 campaign='Y3A1_COADD', sources=None):

        self.medsconf = medsconf
        self.tilename = tilename
        self.band = band

        self.source_dir = files.get_source_dir(
            self.medsconf,
            self.tilename,
            self.band)

        self.campaign = campaign.upper()
        self.sources = sources

        if self.campaign == 'Y3A1_COADD':
            self.finalcut_campaign = 'Y3A1_FINALCUT'
        elif self.campaign == 'Y3A2_COADD':
            self.finalcut_campaign = 'Y3A1_FINALCUT'
        else:
            raise ValueError("determine finalcut campaign "
                             "for '%s'" % self.campaign)

    def get_info(self):
        """Get the SE info for the specified tilename and band.

        Returns
        -------
        info : list of dicts
            A list of dictionaries with the SE source information.
        """

        if hasattr(self, '_info_list'):
            info_list = self._info_list
        else:
            info_list = self._do_query()

            # add full path info
            self._add_full_paths(info_list)

            self._info_list = info_list

        return info_list

    def _do_query(self):
        """Get the SE info for the specified tilename and band.
        """

        conn = self.get_conn()

        query = _QUERY_COADD_SRC_BYTILE % {
            'campaign': self.campaign,
            'finalcut_campaign': self.finalcut_campaign,
            'tilename': self.tilename,
            'band': self.band}
        print('executing query:\n', query)

        curs = conn.cursor()
        curs.execute(query)

        info_list = []

        for row in curs:
            tile, path, fname, comp, band, pai, magzp = row
            info = {
                'tilename': tile,
                'filename': fname,
                'compression': comp,
                'path': path,
                'band': band,
                'pfw_attempt_id': pai,
                'magzp': magzp,
            }

            info_list.append(info)

        return info_list

    def _add_full_paths(self, info_list):
        """Add the full paths to the SE source info list.

        Note that seg maps have .fz for finalcut.
        """

        for info in info_list:

            dirdict = self._get_all_dirs(info)

            info['image_path'] = os.path.join(
                dirdict['image']['local_dir'],
                info['filename']+info['compression'],
            )

            info['bkg_path'] = os.path.join(
                dirdict['bkg']['local_dir'],
                info['filename'].replace(
                    'immasked.fits', 'bkg.fits') + info['compression'],
            )

            info['seg_path'] = os.path.join(
                dirdict['seg']['local_dir'],
                info['filename'].replace(
                    'immasked.fits', 'segmap.fits') + info['compression'],
            )

            info['psf_path'] = os.path.join(
                dirdict['psf']['local_dir'],
                info['filename'].replace('immasked.fits', 'psfexcat.psf'),
            )

    def _get_all_dirs(self, info):
        dirs = {}

        path = info['path']
        dirs['image'] = self._get_dirs(path)
        dirs['seg'] = self._get_dirs(path, type='seg')
        dirs['bkg'] = self._get_dirs(path, type='bkg')
        dirs['psf'] = self._get_dirs(path, type='psf')
        return dirs

    def _get_dirs(self, path, type=None):
        """Get directories for downloads given an input path.

        Parameters
        ----------
        path : str
            The input path, e.g.
            "OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/coadd/".
        type : str, optional
            The type of path. For example, if the input path is

                OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/coadd/

            then setting `type` would yield

                OPS/multiepoch/Y3A1/r2577/DES0215-0458/p01/{type}/

        Returns
        -------
        paths : dict
            Dictionary with keys
                'local_dir': the local directory
                'remote_dir': the remote directory
        """
        local_dir = '%s/%s' % (self.source_dir, path)
        remote_dir = '$DESREMOTE_RSYNC/%s' % path

        local_dir = os.path.expandvars(local_dir)
        remote_dir = os.path.expandvars(remote_dir)

        if type is not None:
            local_dir = self._extract_alt_dir(local_dir, type)
            remote_dir = self._extract_alt_dir(remote_dir, type)

        return {
            'local_dir': local_dir,
            'remote_dir': remote_dir}

    def _extract_alt_dir(self, path, type):
        """
        extract the catalog path from an image path, e.g.

        OPS/finalcut/Y2A1v3/20161124-r2747/D00596130/p01/red/immask/

        would yield

        OPS/finalcut/Y2A1v3/20161124-r2747/D00596130/p01/red/bkg/
        OPS/finalcut/Y2A1v3/20161124-r2747/D00596130/p01/seg
        """

        ps = path.split('/')

        assert ps[-1] == 'immask'

        if type == 'bkg':
            ps[-1] = type
        elif type in ['seg', 'psf']:
            ps = ps[0:-1]
            assert ps[-1] == 'red'
            ps[-1] = type

        return '/'.join(ps)

    def get_conn(self):
        if not hasattr(self, '_conn'):
            self._make_conn()

        return self._conn

    def _make_conn(self):
        import easyaccess as ea
        conn = ea.connect(section='desoper')
        self._conn = conn
