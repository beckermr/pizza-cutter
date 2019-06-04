import tempfile
import os
import numpy as np


class MemMappedNoiseImage(object):
    """A memory mapped image on disk of noise.

    Parameters
    ----------
    seed : int
        The seed for the RNG.
    weight : array-like
        The weight map for the noise field.
    fill_weight : float, optional
        Value used to fill zero-weight pixels.
    sx : int, optional
        Size of patches to generate the noise in the x direction.
    sy : int, optional
        Size of patches to generate the noise in the y direction.
    dir: string
        Optional directory to hold the file
    """

    def __init__(self, *, seed, weight,
                 fill_weight=None, sx=1000, sy=1000,
                 dir=None):

        rng = np.random.RandomState(seed=seed)

        # generate in chunks so that we don't use a ton of memory
        try:
            # if it is a FITS hdu, then this is right
            nx, ny = weight.get_dims()
        except Exception:
            # try a numpy array otherwise
            nx, ny = weight.shape
        self.shape = (nx, ny)

        n_sx = nx // sx
        if n_sx * sx < nx:
            n_sx += 1

        n_sy = ny // sy
        if n_sy * sy < ny:
            n_sy += 1

        if dir is None:
            tmpdir = tempfile.TemporaryDirectory()
            dir = tmpdir.name

        self._fname = tempfile.mktemp(dir=dir, suffix='.dat')

        self._noise = np.memmap(
            self._fname,
            dtype=np.float32,
            mode='w+',
            shape=(nx, ny),
            order='C')

        for isx in range(n_sx):
            for isy in range(n_sy):
                xl = isx * sx
                xu = min(xl + sx, nx)

                yl = isy * sy
                yu = min(yl + sy, ny)

                wgt = weight[xl:xu, yl:yu]
                if fill_weight is not None:
                    zmsk = wgt <= 0
                    zwgt = wgt * (~zmsk) + fill_weight * zmsk
                else:
                    zwgt = wgt
                self._noise[xl:xu, yl:yu] = rng.normal(
                    size=(xu-xl, yu-yl)) * np.sqrt(1.0 / zwgt)
        self._noise.flush()

    def __getitem__(self, slices):
        return self._noise[slices]

    def get_dims(self):
        return self.shape
