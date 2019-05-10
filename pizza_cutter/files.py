import os
import logging
import shutil

logger = logging.getLogger(__name__)


def expandpath(path):
    """Expand environment variables, user home directories (~), and convert
    to an absolute path.
    """
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    path = os.path.realpath(path)
    return path


def makedir_fromfile(fname):
    """Extract the directory and make it if it does not exist."""
    dname = os.path.dirname(fname)
    try_makedir(dname)


def try_makedir(dir):
    """Try to make the directory."""
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

            logger.info(
                "staging out '%s' -> '%s'", self.path, self.final_path)
            shutil.move(self.path, self.final_path)

        self.was_staged_out = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stage_out()
